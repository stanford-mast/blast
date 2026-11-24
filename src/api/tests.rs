use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::Result;
use async_trait::async_trait;

use super::{error::ApiError, handlers::*, runs::handle_run, types::ForkRequest};
use crate::{
    backend::{Exec, RegistryAuth, Resources, RunOutput, VmBackend, VmHandle},
    snapshot::{SnapshotStore, Storage},
    store::{RunStore, Store, VmRecord, VmState},
    worker::{client::ControlPlane, records::Records},
};

const STATE_FILE: &str = "state";

/// Snapshots and forks take a while, and forks report how many overlapped.
/// A stopped machine cannot be checkpointed, as after a host restart.
#[derive(Default)]
struct SlowBackend {
    snapshots: AtomicUsize,
    forking: AtomicUsize,
    most_forking: AtomicUsize,
    stopped: AtomicBool,
    deleted: AtomicUsize,
}

#[async_trait]
impl VmBackend for SlowBackend {
    async fn fork_image(
        &self,
        _image: &str,
        _resources: &Resources,
        _registry_auth: Option<&RegistryAuth>,
    ) -> Result<VmHandle> {
        Ok(VmHandle {
            id: ulid::Ulid::new().to_string(),
        })
    }

    async fn fork_snapshot(&self, snap_dir: &Path, _resources: &Resources) -> Result<VmHandle> {
        let forking = self.forking.fetch_add(1, Ordering::SeqCst) + 1;
        self.most_forking.fetch_max(forking, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(200)).await;
        let state = tokio::fs::read(snap_dir.join(STATE_FILE)).await;
        self.forking.fetch_sub(1, Ordering::SeqCst);
        state?;
        Ok(VmHandle {
            id: ulid::Ulid::new().to_string(),
        })
    }

    async fn run(&self, _vm: &VmHandle, exec: Exec<'_>) -> Result<RunOutput> {
        // A run arrives under its supervisor, and its stop as a command of
        // its own; neither is a shell this backend has.
        if exec.command.contains(".stopped") {
            let mut out = RunOutput::cancelled();
            out.ended = crate::backend::Ended::Exited(0);
            return Ok(out);
        }
        let echoed = exec.command.split_once("'echo ").map(|(_, rest)| rest);
        if let Some(text) = echoed.and_then(|rest| rest.split('\'').next()) {
            let mut out = RunOutput::cancelled();
            out.stdout.bytes = format!("{text}\n").into_bytes();
            out.ended = crate::backend::Ended::Exited(0);
            return Ok(out);
        }
        // Any other command runs until it is cancelled or its machine is deleted.
        while self.deleted.load(Ordering::SeqCst) == 0 {
            if exec.cancel.is_cancelled() {
                return Ok(RunOutput::cancelled());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        anyhow::bail!("machine was deleted")
    }

    async fn pause(&self, _vm: &VmHandle) -> Result<()> {
        Ok(())
    }

    async fn unpause(&self, _vm: &VmHandle) -> Result<()> {
        Ok(())
    }

    async fn snapshot(&self, _vm: &VmHandle, snap_dir: &Path) -> Result<()> {
        if self.stopped.load(Ordering::SeqCst) {
            anyhow::bail!("machine must be running");
        }
        self.snapshots.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(200)).await;
        tokio::fs::create_dir_all(snap_dir).await?;
        tokio::fs::write(snap_dir.join(STATE_FILE), b"memory").await?;
        Ok(())
    }

    async fn suspend(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()> {
        self.snapshot(vm, snap_dir).await
    }

    async fn resume(&self, snap_dir: &Path, resources: &Resources) -> Result<VmHandle> {
        self.fork_snapshot(snap_dir, resources).await
    }

    async fn delete(&self, _vm: &VmHandle) -> Result<()> {
        self.deleted.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn can_recover(&self, _vm: &VmHandle) -> Result<bool> {
        Ok(!self.stopped.load(Ordering::SeqCst))
    }

    async fn is_stopped(&self, _vm: &VmHandle) -> Result<bool> {
        Ok(self.stopped.load(Ordering::SeqCst))
    }

    fn kind(&self) -> &'static str {
        "slow"
    }
}

fn app(data: &Path, backend: Arc<SlowBackend>) -> AppState {
    let control_plane = ControlPlane::local(data).unwrap();
    let records = Records::new(data, control_plane.clone()).unwrap();
    tokio::spawn(records.clone().send_loop());
    let config: crate::config::Config =
        serde_json::from_value(serde_json::json!({ "data_dir": data })).unwrap();
    AppState {
        store: Store::durable(data, 0).unwrap(),
        runs: RunStore::durable(data).unwrap(),
        backend,
        snapshots: Arc::new(SnapshotStore::new(data, Storage::None)),
        config: Arc::new(config),
        total_resources: Resources {
            vcpu: 0,
            memory_mib: 0,
            disk_mib: 0,
        },
        platform: "test".into(),
        pool_id: None,
        draining: Arc::new(AtomicBool::new(false)),
        lifecycle_gate: Arc::default(),
        vm_locks: Arc::default(),
        uploads: Arc::default(),
        run_cancels: Arc::default(),
        checkpoint_uploads: Arc::default(),
        records,
        control_plane,
        allow_local_snapshot_adoption: false,
    }
}

async fn add_source(s: &AppState, handle: Option<VmHandle>) -> String {
    let vm_id = ulid::Ulid::new().to_string();
    let state = if handle.is_some() {
        VmState::Running
    } else {
        VmState::Suspended
    };
    let mut record = VmRecord::new(
        vm_id.clone(),
        state,
        Resources {
            vcpu: 1,
            memory_mib: 256,
            disk_mib: 1024,
        },
    );
    record.handle = handle;
    s.store.insert(record).await;
    vm_id
}

async fn fork_from(s: &AppState, source: &str) -> Result<(), super::error::ApiError> {
    let request = ForkRequest {
        source_vm_id: Some(source.to_owned()),
        ..ForkRequest::default()
    };
    handle_fork(s, None, request).await.map(drop)
}

async fn fork_concurrently(s: &AppState, source: &str, count: usize) {
    let forks: Vec<_> = (0..count)
        .map(|_| {
            let (s, source) = (s.clone(), source.to_owned());
            tokio::spawn(async move { fork_from(&s, &source).await })
        })
        .collect();
    for fork in forks {
        fork.await.unwrap().unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_forks_of_a_running_vm_share_a_checkpoint() {
    let data = tempfile::tempdir().unwrap();
    let backend = Arc::new(SlowBackend::default());
    let s = app(data.path(), backend.clone());
    let source = add_source(&s, Some(VmHandle { id: "src".into() })).await;

    fork_concurrently(&s, &source, 4).await;

    // The first fork's checkpoint may predate the others' arrival; every
    // later fork shares one taken after it arrived.
    assert!(backend.snapshots.load(Ordering::SeqCst) <= 2);
    assert!(backend.most_forking.load(Ordering::SeqCst) > 1);
    let long_ago = std::time::Instant::now()
        .checked_sub(Duration::from_secs(60))
        .unwrap();
    assert!(s
        .snapshots
        .fork_checkpoint_since(&source, long_ago)
        .is_none());
    assert!(!data
        .path()
        .join("transient-snapshots")
        .read_dir()
        .unwrap()
        .any(|_| true));
}

#[tokio::test]
async fn a_fork_shares_only_a_checkpoint_begun_after_it_arrived() {
    let data = tempfile::tempdir().unwrap();
    let backend = Arc::new(SlowBackend::default());
    let s = app(data.path(), backend.clone());
    let source = add_source(&s, Some(VmHandle { id: "src".into() })).await;
    let checkpoint = Arc::new(s.snapshots.transient_snapshot(&source).await.unwrap());
    tokio::fs::write(checkpoint.path().join(STATE_FILE), b"memory")
        .await
        .unwrap();
    let share = |started| {
        s.snapshots
            .share_fork_checkpoint(&source, started, &checkpoint);
    };

    share(
        std::time::Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap(),
    );
    fork_from(&s, &source).await.unwrap();
    assert_eq!(backend.snapshots.load(Ordering::SeqCst), 1);

    share(std::time::Instant::now() + Duration::from_secs(1));
    fork_from(&s, &source).await.unwrap();
    assert_eq!(backend.snapshots.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_forks_of_a_suspended_vm_read_pinned_copies() {
    let data = tempfile::tempdir().unwrap();
    let backend = Arc::new(SlowBackend::default());
    let s = app(data.path(), backend.clone());
    let source = add_source(&s, None).await;
    let snap_dir = s.snapshots.snap_dir(&source);
    tokio::fs::create_dir_all(&snap_dir).await.unwrap();
    tokio::fs::write(snap_dir.join(STATE_FILE), b"memory")
        .await
        .unwrap();
    s.snapshots
        .write_backend_tag(&source, "slow")
        .await
        .unwrap();

    let forks = {
        let (s, source) = (s.clone(), source.clone());
        tokio::spawn(async move { fork_concurrently(&s, &source, 4).await })
    };
    // Deleting the source while forks read its snapshot leaves their pins.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let source_lock = lock_vm(&s, &source).await;
    crate::lifecycle::destroy_vm(&s, &source).await.unwrap();
    drop(source_lock);
    forks.await.unwrap();

    assert_eq!(backend.snapshots.load(Ordering::SeqCst), 0);
    assert!(backend.most_forking.load(Ordering::SeqCst) > 1);
    assert!(!snap_dir.exists());
}

async fn write_local_snapshot(s: &AppState, vm_id: &str) {
    let snap_dir = s.snapshots.snap_dir(vm_id);
    tokio::fs::create_dir_all(&snap_dir).await.unwrap();
    tokio::fs::write(snap_dir.join(STATE_FILE), b"memory")
        .await
        .unwrap();
    s.snapshots.write_backend_tag(vm_id, "slow").await.unwrap();
}

/// What a worker recorded about a VM whose machine was running.
async fn running_manifest(s: &AppState, vm_id: &str) -> crate::snapshot::VmManifest {
    let mut record = VmRecord::new(
        vm_id.to_owned(),
        VmState::Running,
        Resources {
            vcpu: 1,
            memory_mib: 256,
            disk_mib: 1024,
        },
    );
    record.handle = Some(VmHandle {
        id: "machine".into(),
    });
    let manifest = crate::lifecycle::manifest(&record);
    s.snapshots.write_manifest(&manifest).await.unwrap();
    manifest
}

#[tokio::test]
async fn a_running_machine_is_reattached_after_a_restart() {
    let data = tempfile::tempdir().unwrap();
    let s = app(data.path(), Arc::default());
    let manifest = running_manifest(&s, "vm_a").await;

    let record = record_from_manifest(&s, manifest).await.unwrap().unwrap();

    assert_eq!(record.state, VmState::Running);
    assert_eq!(record.handle.unwrap().id, "machine");
}

#[tokio::test]
async fn a_stopped_machine_resumes_from_its_last_snapshot_after_a_restart() {
    let data = tempfile::tempdir().unwrap();
    let backend = Arc::new(SlowBackend::default());
    backend.stopped.store(true, Ordering::SeqCst);
    let s = app(data.path(), backend.clone());
    write_local_snapshot(&s, "vm_a").await;
    let manifest = running_manifest(&s, "vm_a").await;

    let record = record_from_manifest(&s, manifest).await.unwrap().unwrap();

    assert_eq!(record.state, VmState::Suspended);
    assert!(record.handle.is_none());
    assert_eq!(record.snapshot_dir, Some(s.snapshots.snap_dir("vm_a")));
    assert!(!record.snapshot_stored);
    assert_eq!(backend.deleted.load(Ordering::SeqCst), 1);
    let owned = s.snapshots.owned_manifests().await.unwrap();
    assert_eq!(owned[0].backend_handle, None);
}

#[tokio::test]
async fn a_stopped_machine_without_a_snapshot_is_forgotten_after_a_restart() {
    let data = tempfile::tempdir().unwrap();
    let backend = Arc::new(SlowBackend::default());
    backend.stopped.store(true, Ordering::SeqCst);
    let s = app(data.path(), backend);
    let manifest = running_manifest(&s, "vm_a").await;

    assert!(record_from_manifest(&s, manifest).await.unwrap().is_none());
    assert_eq!(s.snapshots.owned_manifests().await.unwrap().len(), 0);
    assert!(!s.snapshots.snap_dir("vm_a").exists());
}

#[tokio::test]
async fn forking_a_vm_whose_machine_stopped_uses_its_last_snapshot() {
    let data = tempfile::tempdir().unwrap();
    let backend = Arc::new(SlowBackend::default());
    let s = app(data.path(), backend.clone());
    let source = add_source(&s, Some(VmHandle { id: "src".into() })).await;
    write_local_snapshot(&s, &source).await;
    backend.stopped.store(true, Ordering::SeqCst);

    fork_from(&s, &source).await.unwrap();

    let source = s.store.get(&source).await.unwrap();
    assert_eq!(source.state, VmState::Suspended);
    assert!(source.handle.is_none());
}

#[tokio::test]
async fn a_vm_whose_machine_stopped_without_a_snapshot_is_deleted() {
    let data = tempfile::tempdir().unwrap();
    let backend = Arc::new(SlowBackend::default());
    let s = app(data.path(), backend.clone());
    let source = add_source(&s, Some(VmHandle { id: "src".into() })).await;
    backend.stopped.store(true, Ordering::SeqCst);

    crate::lifecycle::hand_back(&s, Duration::from_secs(1), false)
        .await
        .unwrap();

    assert!(s.store.get(&source).await.is_none());
    assert_eq!(
        fork_from(&s, &source).await.unwrap_err().status(),
        axum::http::StatusCode::NOT_FOUND
    );
}

fn run_request(time_to_background: u64) -> super::types::RunRequest {
    serde_json::from_value(serde_json::json!({
        "command": "sleep infinity",
        "time_to_background": time_to_background,
    }))
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deleting_a_vm_ends_its_runs() {
    let data = tempfile::tempdir().unwrap();
    let backend = Arc::new(SlowBackend::default());
    let s = app(data.path(), backend.clone());
    let vm_id = add_source(&s, Some(VmHandle { id: "m".into() })).await;
    let (_, background) = handle_run(&s, &vm_id, run_request(0), None).await.unwrap();
    let background = serde_json::to_value(&background.0).unwrap()["run_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let waiting = {
        let (s, vm_id) = (s.clone(), vm_id.clone());
        tokio::spawn(async move { handle_run(&s, &vm_id, run_request(300), None).await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;

    let deleted = delete_vm(
        axum::extract::State(s.clone()),
        axum::extract::Path(vm_id.clone()),
    )
    .await
    .unwrap();
    assert!(deleted.0.deleted);

    assert_eq!(
        waiting.await.unwrap().unwrap_err().status(),
        axum::http::StatusCode::NOT_FOUND
    );
    assert!(s.store.get(&vm_id).await.is_none());
    assert_eq!(s.runs.running_count().await, 0);
    assert!(s.records.flush(Duration::from_secs(5)).await);
    let events = std::fs::read_to_string(data.path().join("control-plane/events.jsonl")).unwrap();
    let failed: Vec<serde_json::Value> = events
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .filter(|event: &serde_json::Value| event["state"] == "failed")
        .collect();
    assert_eq!(failed.len(), 2, "{events}");
    assert!(failed
        .iter()
        .any(|event| event["run_id"] == background.as_str()));
    assert!(failed.iter().all(|event| event["error_message"]
        .as_str()
        .unwrap()
        .contains("was deleted while this run was in flight")));

    let refused = handle_run(&s, &vm_id, run_request(0), None)
        .await
        .unwrap_err();
    assert_eq!(refused.status(), axum::http::StatusCode::NOT_FOUND);
}

fn run_body(body: serde_json::Value) -> super::types::RunRequest {
    serde_json::from_value(body).unwrap()
}

fn answer(
    response: Result<
        (
            axum::http::StatusCode,
            axum::Json<super::types::RunResponse>,
        ),
        ApiError,
    >,
) -> (u16, serde_json::Value) {
    let (status, body) = response.unwrap();
    (status.as_u16(), serde_json::to_value(&body.0).unwrap())
}

async fn running_vm(s: &AppState) -> String {
    add_source(s, Some(VmHandle { id: "m".into() })).await
}

#[tokio::test]
async fn a_repeated_key_answers_with_its_run_and_refuses_another_request() {
    let data = tempfile::tempdir().unwrap();
    let s = app(data.path(), Arc::new(SlowBackend::default()));
    let vm_id = running_vm(&s).await;
    let key = |digest: &str| Some(("retry-1".to_owned(), digest.to_owned()));
    let request = || run_body(serde_json::json!({"command": "echo once"}));

    let first = answer(handle_run(&s, &vm_id, request(), key("a")).await);
    let replay = answer(handle_run(&s, &vm_id, request(), key("a")).await);
    let other = handle_run(&s, &vm_id, request(), key("b"))
        .await
        .unwrap_err();

    assert_eq!(first.0, 200);
    assert_eq!(first.1["stdout"], "once\n");
    assert_eq!(replay, first);
    assert_eq!(other.status(), axum::http::StatusCode::CONFLICT);
    assert_eq!(s.runs.of_vm(&vm_id).await.len(), 1);
}

#[tokio::test]
async fn a_request_naming_nothing_wakes_the_vm_and_makes_no_run() {
    let data = tempfile::tempdir().unwrap();
    let s = app(data.path(), Arc::new(SlowBackend::default()));
    let vm_id = running_vm(&s).await;

    let woken = answer(handle_run(&s, &vm_id, run_body(serde_json::json!({})), None).await);
    let refused = handle_run(
        &s,
        &vm_id,
        run_body(serde_json::json!({"command": "", "timeout": 5})),
        None,
    )
    .await
    .unwrap_err();

    assert_eq!(
        woken,
        (
            200,
            serde_json::json!({
                "state": "completed", "exit_code": 0,
                "stdout": "", "stdout_encoding": "utf-8",
                "stderr": "", "stderr_encoding": "utf-8",
            })
        )
    );
    assert_eq!(refused.status(), axum::http::StatusCode::BAD_REQUEST);
    assert!(s.runs.of_vm(&vm_id).await.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_long_run_does_not_hold_up_its_session_and_is_cancelled_once() {
    let data = tempfile::tempdir().unwrap();
    let s = app(data.path(), Arc::new(SlowBackend::default()));
    let vm_id = running_vm(&s).await;

    let long = answer(
        handle_run(
            &s,
            &vm_id,
            run_body(serde_json::json!({"command": "sleep infinity", "time_to_background": 0})),
            None,
        )
        .await,
    );
    let quick = answer(
        handle_run(
            &s,
            &vm_id,
            run_body(serde_json::json!({"command": "echo quick"})),
            None,
        )
        .await,
    );

    assert_eq!((long.0, quick.0), (202, 200));
    assert_eq!(long.1["session_id"], quick.1["session_id"]);
    assert_eq!(quick.1["stdout"], "quick\n");

    let long_id = long.1["run_id"].as_str().unwrap().to_owned();
    let cancel = |run_id: String| {
        super::runs::cancel_run(
            axum::extract::State(s.clone()),
            axum::extract::Path((vm_id.clone(), run_id)),
        )
    };
    assert!(cancel(long_id.clone()).await.unwrap().0.cancelled);
    assert!(!cancel(long_id.clone()).await.unwrap().0.cancelled);

    let cancelled = s.runs.get(&long_id).await.unwrap();
    assert_eq!(cancelled.state, crate::store::RunState::Cancelled);
    assert_eq!(cancelled.exit_code, None);
}

#[test]
fn a_name_that_could_not_be_an_image_is_refused_as_one() {
    for image in [
        "ubuntu:24.04",
        "ghcr.io/owner/repo:tag",
        "registry.example:5000/a/b@sha256:0123abcd",
    ] {
        assert!(super::handlers::is_image_reference(image), "{image}");
    }
    for image in ["", "not a valid/image::ref", "ubuntu:24.04\n", "repo;rm"] {
        assert!(!super::handlers::is_image_reference(image), "{image:?}");
    }
}

#[test]
fn an_image_is_named_in_full() {
    for (image, full) in [
        ("alpine:3.20", "docker.io/library/alpine:3.20"),
        ("grafana/grafana", "docker.io/grafana/grafana"),
        ("quay.io/podman/stable:v5.6", "quay.io/podman/stable:v5.6"),
        ("localhost:5000/app", "localhost:5000/app"),
    ] {
        assert_eq!(super::handlers::canonical_image(image), full);
    }
}

#[test]
fn a_deletion_the_control_plane_did_not_take_can_be_retried() {
    use crate::worker::client::CallError;
    use axum::response::IntoResponse;

    let unreachable = anyhow::Error::new(CallError::Transient(anyhow::anyhow!("reset")));
    assert_eq!(
        super::handlers::deletion_error(unreachable)
            .into_response()
            .status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        super::handlers::deletion_error(anyhow::anyhow!("disk"))
            .into_response()
            .status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fork_its_caller_stopped_waiting_for_still_makes_its_vm() {
    use tower::ServiceExt;

    let data = tempfile::tempdir().unwrap();
    let mut s = app(data.path(), Arc::new(SlowBackend::default()));
    // A declared pool: the fork reserves its VM before it starts it.
    s.total_resources = Resources {
        vcpu: 4,
        memory_mib: 4096,
        disk_mib: 8192,
    };
    let source = add_source(&s, Some(VmHandle { id: "src".into() })).await;
    let vm_id = ulid::Ulid::new().to_string();
    let request = axum::http::Request::post("/v1/fork")
        .header("content-type", "application/json")
        .header(VM_ID_HEADER, &vm_id)
        .body(axum::body::Body::from(
            serde_json::json!({ "source_vm_id": source }).to_string(),
        ))
        .unwrap();

    let fork = super::router(s.clone(), None).oneshot(request);
    assert!(tokio::time::timeout(Duration::from_millis(50), fork)
        .await
        .is_err());

    let started = async {
        while s.store.get(&vm_id).await.and_then(|vm| vm.handle).is_none() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), started)
        .await
        .expect("the abandoned fork's VM never started");
}

/// A connected worker's loopback listener with `leave` mounted, whose leave
/// requests are each answered "left".
fn connected_listener(s: AppState, token: Option<&str>) -> axum::Router {
    let (requests, mut asked) = tokio::sync::mpsc::channel::<tokio::sync::oneshot::Sender<bool>>(1);
    tokio::spawn(async move {
        while let Some(left) = asked.recv().await {
            let _ = left.send(true);
        }
    });
    super::diagnostics_router(
        s,
        token.map(Into::into),
        crate::worker::leave::route(requests),
    )
}

async fn ask_to_leave(router: axum::Router, token: Option<&str>) -> axum::http::StatusCode {
    use tower::ServiceExt;

    let mut request = axum::http::Request::post(crate::worker::leave::PATH);
    if let Some(token) = token {
        request = request.header("authorization", format!("Bearer {token}"));
    }
    let request = request.body(axum::body::Body::empty()).unwrap();
    router.oneshot(request).await.unwrap().status()
}

#[tokio::test]
async fn a_connected_worker_without_a_token_leaves_when_asked_on_loopback() {
    let data = tempfile::tempdir().unwrap();
    let s = app(data.path(), Arc::new(SlowBackend::default()));
    assert_eq!(
        ask_to_leave(connected_listener(s, None), None).await,
        axum::http::StatusCode::NO_CONTENT
    );
}

#[tokio::test]
async fn a_connected_worker_with_a_token_leaves_only_for_its_token() {
    let data = tempfile::tempdir().unwrap();
    let s = app(data.path(), Arc::new(SlowBackend::default()));
    assert_eq!(
        ask_to_leave(connected_listener(s.clone(), Some("secret")), None).await,
        axum::http::StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        ask_to_leave(connected_listener(s, Some("secret")), Some("secret")).await,
        axum::http::StatusCode::NO_CONTENT
    );
}

#[tokio::test]
async fn a_worker_outside_a_pool_serves_no_leave() {
    let data = tempfile::tempdir().unwrap();
    let s = app(data.path(), Arc::new(SlowBackend::default()));
    assert_eq!(
        ask_to_leave(super::router(s, None), None).await,
        axum::http::StatusCode::NOT_FOUND
    );
}
