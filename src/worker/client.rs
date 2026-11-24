//! HTTP calls to the control plane, all under this worker's own path
//! `{endpoint}/api/v1/pools/{pool_id}/workers/{worker_id}` and all
//! authenticated with the pool's join key.

use std::{io::Write, path::PathBuf, time::Duration};

use opentelemetry_proto::tonic::{
    collector::logs::v1::ExportLogsServiceRequest,
    common::v1::{any_value, AnyValue, KeyValue},
    logs::v1::{LogRecord, ResourceLogs, ScopeLogs},
    resource::v1::Resource,
};
use prost::Message;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use crate::config::RemoteConfig;

use super::{
    heartbeat::{WorkerHeartbeat, WorkerHeartbeatResponse},
    records::Activity,
};

const MAX_LOCAL_EVENT_LOG_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone)]
pub struct ControlPlane {
    target: Target,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotReservation {
    pub vm_generation: i64,
    pub snapshot_id: String,
    #[serde(default = "default_snapshot_lease_renew_after_secs")]
    pub lease_renew_after_secs: u64,
}

#[derive(Debug, Serialize)]
struct ReconcileVmsRequest<'a> {
    vm_ids: &'a [String],
}

#[derive(Debug, Deserialize)]
struct ReconcileVmsResponse {
    retained_vm_ids: Vec<String>,
}

const fn default_snapshot_lease_renew_after_secs() -> u64 {
    0
}

#[derive(Clone)]
enum Target {
    Local(PathBuf),
    Remote(Remote),
}

#[derive(Clone)]
struct Remote {
    http: reqwest::Client,
    endpoint: String,
    worker_url: String,
    worker_id: String,
    incarnation: String,
    api_key: String,
    ready: tokio::sync::watch::Sender<bool>,
}

/// Why a call failed: worth retrying, or refused for good.
#[derive(Debug, thiserror::Error)]
pub enum CallError {
    /// The control plane was unreachable or answered with a transient
    /// status (5xx, 408, 429).
    #[error("{0:#}")]
    Transient(anyhow::Error),
    /// The control plane refused the call; repeating it will not help.
    #[error("rejected: {status}: {body}")]
    Rejected { status: StatusCode, body: String },
}

impl ControlPlane {
    pub fn incarnation(&self) -> Option<&str> {
        match &self.target {
            Target::Local(_) => None,
            Target::Remote(remote) => Some(&remote.incarnation),
        }
    }

    pub fn local(data_dir: &std::path::Path) -> anyhow::Result<Self> {
        let root = data_dir.join("control-plane");
        std::fs::create_dir_all(&root)?;
        for entry in std::fs::read_dir(&root)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().starts_with("events__") {
                std::fs::remove_file(entry.path())?;
            }
        }
        Ok(Self {
            target: Target::Local(root),
        })
    }

    pub fn new(cfg: &RemoteConfig, worker_id: &str, incarnation: &str) -> Self {
        let (ready, _) = tokio::sync::watch::channel(false);
        Self {
            target: Target::Remote(Remote {
                http: reqwest::Client::builder()
                    .timeout(Duration::from_secs(30))
                    .build()
                    .expect("reqwest client"),
                endpoint: cfg.endpoint.clone(),
                worker_url: format!(
                    "{}/api/v1/pools/{}/workers/{worker_id}",
                    cfg.endpoint, cfg.pool_id
                ),
                worker_id: worker_id.to_owned(),
                incarnation: incarnation.to_owned(),
                api_key: cfg.api_key.clone(),
                ready,
            }),
        }
    }

    pub fn allocate_vm_id(&self, allocated: Option<String>) -> anyhow::Result<String> {
        match (&self.target, allocated) {
            (_, Some(vm_id)) => Ok(vm_id),
            (Target::Local(_), None) => Ok(crate::store::new_vm_id()),
            (Target::Remote(_), None) => anyhow::bail!("the control plane must allocate the VM id"),
        }
    }

    pub async fn heartbeat(
        &self,
        body: &WorkerHeartbeat,
    ) -> Result<WorkerHeartbeatResponse, CallError> {
        let Target::Remote(remote) = &self.target else {
            return Err(CallError::Rejected {
                status: StatusCode::BAD_REQUEST,
                body: "a local control plane does not heartbeat".into(),
            });
        };
        let request = self
            .remote_request(remote.http.put(&remote.worker_url))
            .query(&[("incarnation", &remote.incarnation)])
            .json(body);
        let response = self.send(request).await?;
        let response = response.json().await.map_err(|e| {
            CallError::Transient(anyhow::Error::new(e).context("heartbeat response"))
        })?;
        Ok(response)
    }

    pub fn mark_ready(&self) {
        if let Target::Remote(remote) = &self.target {
            remote.ready.send_replace(true);
        }
    }

    pub async fn ready(&self) {
        let Target::Remote(remote) = &self.target else {
            return;
        };
        let mut ready = remote.ready.subscribe();
        while !*ready.borrow_and_update() {
            if ready.changed().await.is_err() {
                return;
            }
        }
    }

    /// `PUT {worker}/{path}` with a JSON body.
    pub async fn put(&self, path: &str, body: &(impl Serialize + Sync)) -> Result<(), CallError> {
        match &self.target {
            Target::Local(root) => local_put(root, path, body),
            Target::Remote(remote) => {
                let request = self
                    .remote_request(remote.http.put(format!("{}/{path}", remote.worker_url)))
                    .query(&[("incarnation", &remote.incarnation)])
                    .json(body);
                self.send(request).await.map(drop)
            }
        }
    }

    pub async fn reconcile_vms(&self, vm_ids: &[String]) -> Result<Vec<String>, CallError> {
        let Target::Remote(remote) = &self.target else {
            return Ok(vm_ids.to_vec());
        };
        let request = self
            .remote_request(
                remote
                    .http
                    .post(format!("{}/vms/reconcile", remote.worker_url)),
            )
            .query(&[("incarnation", &remote.incarnation)])
            .json(&ReconcileVmsRequest { vm_ids });
        self.send(request)
            .await?
            .json::<ReconcileVmsResponse>()
            .await
            .map(|response| response.retained_vm_ids)
            .map_err(|error| {
                CallError::Transient(
                    anyhow::Error::new(error).context("decode VM reconciliation response"),
                )
            })
    }

    pub async fn reserve_snapshot(&self, vm_id: &str) -> Result<SnapshotReservation, CallError> {
        match &self.target {
            Target::Local(_) => Ok(SnapshotReservation {
                vm_generation: 1,
                snapshot_id: ulid::Ulid::new().to_string(),
                lease_renew_after_secs: 0,
            }),
            Target::Remote(remote) => {
                let request = self
                    .remote_request(
                        remote
                            .http
                            .post(format!("{}/vms/{vm_id}/snapshots", remote.worker_url)),
                    )
                    .query(&[("incarnation", &remote.incarnation)]);
                self.send(request).await?.json().await.map_err(|error| {
                    CallError::Transient(
                        anyhow::Error::new(error).context("decode snapshot reservation"),
                    )
                })
            }
        }
    }

    pub async fn renew_snapshot(
        &self,
        vm_id: &str,
        reservation: &SnapshotReservation,
    ) -> Result<(), CallError> {
        match &self.target {
            Target::Local(_) => Ok(()),
            Target::Remote(remote) => {
                let vm_generation = reservation.vm_generation.to_string();
                let request = self
                    .remote_request(remote.http.patch(format!(
                        "{}/vms/{vm_id}/snapshots/{}",
                        remote.worker_url, reservation.snapshot_id
                    )))
                    .query(&[
                        ("incarnation", remote.incarnation.as_str()),
                        ("vm_generation", vm_generation.as_str()),
                    ]);
                self.send(request).await.map(drop)
            }
        }
    }

    pub async fn commit_snapshot(
        &self,
        vm_id: &str,
        reservation: &SnapshotReservation,
    ) -> Result<(), CallError> {
        match &self.target {
            Target::Local(root) => local_put(root, &format!("vms/{vm_id}/snapshot"), reservation),
            Target::Remote(remote) => {
                let vm_generation = reservation.vm_generation.to_string();
                let request = self
                    .remote_request(remote.http.put(format!(
                        "{}/vms/{vm_id}/snapshots/{}",
                        remote.worker_url, reservation.snapshot_id
                    )))
                    .query(&[
                        ("incarnation", remote.incarnation.as_str()),
                        ("vm_generation", vm_generation.as_str()),
                    ]);
                self.send(request).await.map(drop)
            }
        }
    }

    /// `DELETE {worker}/{path}`.
    pub async fn delete(&self, path: &str) -> Result<(), CallError> {
        match &self.target {
            Target::Local(root) => local_delete(root, path),
            Target::Remote(remote) => {
                let request = self
                    .remote_request(remote.http.delete(format!("{}/{path}", remote.worker_url)))
                    .query(&[("incarnation", &remote.incarnation)]);
                match self.send(request).await {
                    Ok(_)
                    | Err(CallError::Rejected {
                        status: StatusCode::NOT_FOUND,
                        ..
                    }) => Ok(()),
                    Err(error) => Err(error),
                }
            }
        }
    }

    pub async fn activity(&self, activity: &Activity) -> Result<(), CallError> {
        match &self.target {
            Target::Local(root) => local_event(root, activity),
            Target::Remote(remote) => {
                let body = encode_activity(activity);
                let request = self
                    .remote_request(remote.http.post(format!(
                        "{}/api/v1/internal/store/telemetry/events",
                        remote.endpoint
                    )))
                    .header("content-type", "application/x-protobuf")
                    .body(body);
                let response = self.send(request).await?;
                let receipt: TelemetryReceipt = response.json().await.map_err(|error| {
                    CallError::Transient(
                        anyhow::Error::new(error).context("decode telemetry receipt"),
                    )
                })?;
                if receipt.accepted == 1 && receipt.refused == 0 {
                    Ok(())
                } else {
                    Err(CallError::Transient(anyhow::anyhow!(
                        "telemetry receipt accepted {} and refused {}, expected 1 and 0",
                        receipt.accepted,
                        receipt.refused
                    )))
                }
            }
        }
    }

    fn remote_request(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let Target::Remote(remote) = &self.target else {
            return request;
        };
        request
            .bearer_auth(&remote.api_key)
            .header("control-plane-worker-id", &remote.worker_id)
            .header("control-plane-worker-incarnation", &remote.incarnation)
    }

    async fn send(&self, request: reqwest::RequestBuilder) -> Result<reqwest::Response, CallError> {
        let resp = request
            .send()
            .await
            .map_err(|e| CallError::Transient(e.into()))?;
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let body = resp.text().await.map_err(|error| {
            CallError::Transient(anyhow::Error::new(error).context("read control-plane error"))
        })?;
        if status.is_server_error()
            || status == StatusCode::REQUEST_TIMEOUT
            || status == StatusCode::TOO_MANY_REQUESTS
        {
            return Err(CallError::Transient(anyhow::anyhow!("{status}: {body}")));
        }
        Err(CallError::Rejected { status, body })
    }

    /// The tunnel URL as a WebSocket URL: the control plane may answer with
    /// a `ws(s)`, an `http(s)`, or an endpoint-relative URL.
    pub fn resolve_tunnel_url(&self, tunnel_url: &str) -> String {
        let Target::Remote(remote) = &self.target else {
            return tunnel_url.to_owned();
        };
        let absolute = if tunnel_url.starts_with('/') {
            format!("{}{tunnel_url}", remote.endpoint)
        } else {
            tunnel_url.to_owned()
        };
        for (http, ws) in [("https://", "wss://"), ("http://", "ws://")] {
            if let Some(rest) = absolute.strip_prefix(http) {
                return format!("{ws}{rest}");
            }
        }
        absolute
    }
}

#[derive(Deserialize)]
struct TelemetryReceipt {
    accepted: usize,
    refused: usize,
}

fn local_path(root: &std::path::Path, path: &str) -> PathBuf {
    root.join(format!("{}.json", path.replace('/', "__")))
}

fn local_put(
    root: &std::path::Path,
    path: &str,
    body: &(impl Serialize + Sync),
) -> Result<(), CallError> {
    crate::durable::write_json_atomic(&local_path(root, path), body)
        .map_err(|error| CallError::Transient(error.into()))
}

fn local_delete(root: &std::path::Path, path: &str) -> Result<(), CallError> {
    match std::fs::remove_file(local_path(root, path)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(CallError::Transient(error.into())),
    }?;
    let Some(vm_id) = path
        .strip_prefix("vms/")
        .filter(|value| !value.contains('/'))
    else {
        return Ok(());
    };
    let prefix = format!("vms__{vm_id}__");
    for entry in std::fs::read_dir(root).map_err(|error| CallError::Transient(error.into()))? {
        let entry = entry.map_err(|error| CallError::Transient(error.into()))?;
        if entry.file_name().to_string_lossy().starts_with(&prefix) {
            std::fs::remove_file(entry.path())
                .map_err(|error| CallError::Transient(error.into()))?;
        }
    }
    Ok(())
}

fn local_event(root: &std::path::Path, activity: &Activity) -> Result<(), CallError> {
    let path = root.join("events.jsonl");
    if path
        .metadata()
        .is_ok_and(|metadata| metadata.len() >= MAX_LOCAL_EVENT_LOG_BYTES)
    {
        std::fs::File::create(&path)
            .and_then(|file| file.sync_all())
            .map_err(|error| CallError::Transient(error.into()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|error| CallError::Transient(error.into()))?;
    serde_json::to_writer(&mut file, activity)
        .map_err(|error| CallError::Transient(error.into()))?;
    file.write_all(b"\n")
        .and_then(|()| file.sync_data())
        .map_err(|error| CallError::Transient(error.into()))
}

fn string(key: &str, value: &str) -> KeyValue {
    KeyValue {
        key: key.into(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(value.into())),
        }),
        ..Default::default()
    }
}

fn encode_activity(activity: &Activity) -> Vec<u8> {
    let mut attributes = vec![
        string("event.v", "2"),
        string("action", &activity.action),
        string("state", &activity.state),
        string("action_id", &activity.action_id),
        string("event_id", &activity.event_id),
        string("status_code", &activity.status_code.to_string()),
        string("attributes", "{}"),
    ];
    if let Some(vm_id) = &activity.vm_id {
        attributes.push(string("vm_id", vm_id));
    }
    if let Some(run_id) = &activity.run_id {
        attributes.push(string("run_id", run_id));
    }
    if let Some(exit_code) = activity.exit_code {
        attributes.push(string("exit_code", &exit_code.to_string()));
    }
    if let Some(bytes) = activity.stdout_bytes {
        attributes.push(string("stdout_bytes", &bytes.to_string()));
    }
    if let Some(bytes) = activity.stderr_bytes {
        attributes.push(string("stderr_bytes", &bytes.to_string()));
    }
    if let Some(time) = activity.requested_at_ms {
        attributes.push(string("requested_at_ms", &time.to_string()));
    }
    if let Some(time) = activity.started_at_ms {
        attributes.push(string("started_at_ms", &time.to_string()));
    }
    if let Some(time) = activity.completed_at_ms {
        attributes.push(string("completed_at_ms", &time.to_string()));
    }
    // What the action took: in all, waiting to start, and running.
    let took = |from: Option<u64>, to: Option<u64>| Some(to?.saturating_sub(from?).to_string());
    let began = activity.requested_at_ms.or(activity.started_at_ms);
    if let Some(ms) = took(began, activity.completed_at_ms) {
        attributes.push(string("duration_ms", &ms));
    }
    if let Some(ms) = took(activity.requested_at_ms, activity.started_at_ms) {
        attributes.push(string("queue_latency_ms", &ms));
    }
    if activity.action == "run" {
        if let Some(ms) = took(activity.started_at_ms, activity.completed_at_ms) {
            attributes.push(string("vm_exec_ms", &ms));
        }
    }
    if let Some(error) = &activity.error_message {
        attributes.push(string("error_message", error));
    }
    if let Some(body) = &activity.request_body_content {
        attributes.push(string("request_body_content", body));
        attributes.push(string("request_size_bytes", &body.len().to_string()));
    }
    if let Some(body) = &activity.response_body_content {
        attributes.push(string("response_body_content", body));
        attributes.push(string("response_size_bytes", &body.len().to_string()));
    }
    let request = ExportLogsServiceRequest {
        resource_logs: vec![ResourceLogs {
            resource: Some(Resource::default()),
            scope_logs: vec![ScopeLogs {
                log_records: vec![LogRecord {
                    time_unix_nano: activity.time_unix_nano,
                    body: Some(AnyValue {
                        value: Some(any_value::Value::StringValue("activity.event".into())),
                    }),
                    attributes,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }],
    };
    request.encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_vm_delete_cascades_only_its_control_plane_records() {
        let dir = tempfile::tempdir().unwrap();
        local_put(dir.path(), "vms/vm-a", &serde_json::json!({})).unwrap();
        local_put(dir.path(), "vms/vm-a/runs/run-1", &serde_json::json!({})).unwrap();
        local_put(dir.path(), "vms/vm-a/snapshot", &serde_json::json!({})).unwrap();
        local_put(dir.path(), "vms/vm-ab/runs/run-2", &serde_json::json!({})).unwrap();

        local_delete(dir.path(), "vms/vm-a").unwrap();

        assert!(!local_path(dir.path(), "vms/vm-a").exists());
        assert!(!local_path(dir.path(), "vms/vm-a/runs/run-1").exists());
        assert!(!local_path(dir.path(), "vms/vm-a/snapshot").exists());
        assert!(local_path(dir.path(), "vms/vm-ab/runs/run-2").exists());
    }

    #[test]
    fn an_activity_says_how_long_it_took() {
        let attribute = |action: &str, key: &str| {
            let mut activity = Activity::operation(action, "completed", None);
            activity.requested_at_ms = Some(1_000);
            activity.started_at_ms = Some(1_200);
            activity.completed_at_ms = Some(4_200);
            let request =
                ExportLogsServiceRequest::decode(encode_activity(&activity).as_slice()).unwrap();
            let record = &request.resource_logs[0].scope_logs[0].log_records[0];
            record
                .attributes
                .iter()
                .find(|kv| kv.key == key)
                .and_then(|kv| match kv.value.as_ref()?.value.as_ref()? {
                    any_value::Value::StringValue(value) => Some(value.clone()),
                    _ => None,
                })
        };

        assert_eq!(attribute("run", "duration_ms").as_deref(), Some("3200"));
        assert_eq!(attribute("run", "queue_latency_ms").as_deref(), Some("200"));
        assert_eq!(attribute("run", "vm_exec_ms").as_deref(), Some("3000"));
        assert_eq!(attribute("fork", "duration_ms").as_deref(), Some("3200"));
        assert_eq!(attribute("fork", "vm_exec_ms"), None);
    }

    #[test]
    fn local_activity_log_rotates_at_its_bound() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        std::fs::File::create(&path)
            .unwrap()
            .set_len(MAX_LOCAL_EVENT_LOG_BYTES)
            .unwrap();

        local_event(dir.path(), &Activity::operation("run", "completed", None)).unwrap();

        assert!(path.metadata().unwrap().len() < MAX_LOCAL_EVENT_LOG_BYTES);
    }
}
