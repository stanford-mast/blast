use std::{
    collections::HashMap,
    fmt::Write as _,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    Json,
};

use crate::{
    admission::CapacityError,
    backend::{Ended, Exec, Resources, Stdout, VmBackend, VmHandle},
    config::Config,
    snapshot::{SnapshotStore, TransientSnapshot},
    store::{RunState, RunStore, Store, VmRecord, VmState},
    worker::records::{Activity, Records},
};

use super::{error::ApiError, sync::Uploads, types::*};

#[derive(Clone)]
pub struct AppState {
    pub store: Store,
    pub runs: RunStore,
    pub backend: Arc<dyn VmBackend>,
    pub snapshots: Arc<SnapshotStore>,
    pub config: Arc<Config>,
    /// The resolved pool total `lifecycle::spawn` drives its pressure loop
    /// against -- `{0,0,0}` means no pool configured (unlimited admission).
    pub total_resources: Resources,
    /// `blast-<arch>` unless overridden; see `backend::worker_platform`.
    pub platform: String,
    pub pool_id: Option<String>,
    /// Set once a drain begins: new forks are refused from then on.
    pub draining: Arc<AtomicBool>,
    /// Serializes the start of mutations with the leave/stop quiescence
    /// barrier. Requests that began before draining finish under a read lock;
    /// the barrier takes the write lock before inspecting VM state.
    pub lifecycle_gate: Arc<tokio::sync::RwLock<()>>,
    /// Serializes backend and lifecycle work for each VM without coupling
    /// unrelated VMs. The global lifecycle gate remains only the drain fence.
    pub vm_locks: Arc<std::sync::Mutex<HashMap<String, std::sync::Weak<AsyncMutex<()>>>>>,
    pub uploads: Uploads,
    pub run_cancels: super::runs::RunCancels,
    pub records: Records,
    pub control_plane: crate::worker::client::ControlPlane,
    pub allow_local_snapshot_adoption: bool,
}

pub async fn lock_vm(s: &AppState, vm_id: &str) -> OwnedMutexGuard<()> {
    let lock = vm_lock(s, vm_id);
    lock.lock_owned().await
}

pub fn try_lock_vm(s: &AppState, vm_id: &str) -> Option<OwnedMutexGuard<()>> {
    vm_lock(s, vm_id).try_lock_owned().ok()
}

fn vm_lock(s: &AppState, vm_id: &str) -> Arc<AsyncMutex<()>> {
    let mut locks = s.vm_locks.lock().expect("VM lock map poisoned");
    if let Some(lock) = locks.get(vm_id).and_then(std::sync::Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(AsyncMutex::new(()));
    locks.insert(vm_id.to_owned(), Arc::downgrade(&lock));
    lock
}

pub async fn begin_mutation(
    s: &AppState,
) -> Result<tokio::sync::RwLockReadGuard<'_, ()>, ApiError> {
    let guard = s.lifecycle_gate.read().await;
    if s.draining.load(Ordering::SeqCst) {
        return Err(ApiError::unavailable("worker is draining"));
    }
    Ok(guard)
}

type ApiResult<T> = Result<Json<T>, ApiError>;

/// Header carrying the VM id the control plane minted for a fork, so both sides agree.
pub const VM_ID_HEADER: &str = "x-vm-id";

const DEFAULT_RESOURCES: Resources = Resources {
    vcpu: 2,
    memory_mib: 2048,
    disk_mib: 10240,
};

/// A time as the API writes one: to the second, with a numeric UTC offset.
pub(super) fn rfc3339(t: SystemTime) -> String {
    let time = humantime::format_rfc3339_seconds(t).to_string();
    format!("{}+00:00", time.trim_end_matches('Z'))
}

pub(super) fn vm_not_found() -> ApiError {
    ApiError::not_found("vm")
}

pub fn storage_unavailable(e: &anyhow::Error) -> ApiError {
    tracing::warn!(err = %format!("{e:#}"), "snapshot storage unavailable");
    ApiError::unavailable(format!("snapshot storage: {e:#}"))
}

fn persist_vm_record(records: &Records, record: &VmRecord) -> Result<(), ApiError> {
    records
        .vm(record)
        .map_err(|error| ApiError::unavailable(format!("persist VM record: {error}")))
}

pub async fn find_vm(s: &AppState, vm_id: &str) -> Result<VmRecord, ApiError> {
    if let Some(record) = s.store.get(vm_id).await {
        return Ok(record);
    }
    if !s.allow_local_snapshot_adoption {
        return Err(vm_not_found());
    }
    let Some(manifest) = s
        .snapshots
        .adopt_local_manifest(vm_id)
        .await
        .map_err(|error| ApiError::internal(format!("recover local VM: {error:#}")))?
    else {
        return Err(vm_not_found());
    };
    let Some(record) = record_from_manifest(s, manifest).await? else {
        return Err(vm_not_found());
    };
    s.store.insert(record.clone()).await;
    persist_vm_record(&s.records, &record)?;
    Ok(record)
}

async fn peek_vm(s: &AppState, vm_id: &str) -> Result<VmRecord, ApiError> {
    find_vm(s, vm_id).await
}

/// Rebuilds a VM's record after a restart or a claim: a machine still
/// running is reattached, anything else resumes from its last snapshot.
/// `None` when the VM has neither and is lost; it is then forgotten.
pub async fn record_from_manifest(
    s: &AppState,
    manifest: crate::snapshot::VmManifest,
) -> Result<Option<VmRecord>, ApiError> {
    let vm_id = manifest.vm_id.clone();
    let created_at = manifest.created_at();
    let inspect = |e: anyhow::Error| ApiError::internal(format!("inspect backend VM: {e:#}"));
    let mut stale_handle = None;
    let recovered = match manifest.backend_handle.as_deref() {
        Some(id) => {
            let handle = VmHandle { id: id.to_owned() };
            if s.backend.can_recover(&handle).await.map_err(inspect)? {
                Some(handle)
            } else {
                stale_handle = Some(handle);
                None
            }
        }
        None => None,
    };
    let state = if recovered.is_some() {
        VmState::Running
    } else {
        VmState::Suspended
    };
    let mut record = VmRecord::new(vm_id.clone(), state, manifest.resources);
    record.name = manifest.name;
    record.description = manifest.description;
    record.image = manifest.image;
    record.parent_vm_id = manifest.parent_vm_id;
    record.workdir = manifest.workdir;
    record.created_at = created_at;
    record.handle = recovered;
    if record.handle.is_some() {
        return Ok(Some(record));
    }
    if let Some(handle) = &stale_handle {
        if s.backend.is_stopped(handle).await.map_err(inspect)? {
            if let Err(error) = s.backend.delete(handle).await {
                tracing::warn!(vm_id, err = %format!("{error:#}"), "removing a stopped machine failed");
            }
        }
    }
    let Some(last) = s
        .snapshots
        .last_snapshot(&vm_id)
        .await
        .map_err(|e| ApiError::internal(format!("inspect local snapshot: {e:#}")))?
    else {
        forget_lost_vm(s, &vm_id).await?;
        return Ok(None);
    };
    if stale_handle.is_some() {
        // Without the handle, a later restart restores it too, and its
        // snapshot can be stored.
        s.snapshots
            .write_manifest(&crate::lifecycle::manifest(&record))
            .await
            .map_err(|e| ApiError::internal(format!("record suspended VM: {e:#}")))?;
    }
    record.snapshot_dir = last.local;
    record.snapshot_stored = last.stored;
    record.idle_since = Some(std::time::Instant::now());
    Ok(Some(record))
}

/// Forgets a VM that has neither a machine nor a snapshot left, and reports
/// it deleted once the control plane can hear it.
async fn forget_lost_vm(s: &AppState, vm_id: &str) -> Result<(), ApiError> {
    tracing::warn!(
        vm_id,
        "VM has neither a running machine nor a snapshot; it is lost"
    );
    let forget = |e: anyhow::Error| ApiError::internal(format!("forget lost VM: {e:#}"));
    crate::store::forget_portable_vm_state(vm_id, &s.store, &s.runs)
        .await
        .map_err(forget)?;
    s.snapshots
        .discard_local_vm_state(vm_id)
        .await
        .map_err(forget)?;
    let (records, vm_id) = (s.records.clone(), vm_id.to_owned());
    tokio::spawn(async move {
        if let Err(error) = records.vm_deleted(&vm_id).await {
            tracing::warn!(vm_id, err = %error, "reporting a lost VM deleted failed");
        }
    });
    Ok(())
}

impl From<CapacityError> for ApiError {
    fn from(e: CapacityError) -> Self {
        Self::capacity_unavailable(e.to_string())
    }
}

/// The API's `VmState`: `running` while a run or a sync uses the VM.
pub fn api_state_str(vm: &VmRecord) -> &'static str {
    if vm.in_use() {
        "running"
    } else {
        "idle"
    }
}

pub async fn vm_object(s: &AppState, rec: VmRecord) -> Vm {
    Vm {
        state: api_state_str(&rec),
        sessions: super::runs::session_objects(s, &rec).await,
        created_at: rfc3339(rec.created_at),
        // BLAST has no notion of orgs; the control plane owns the real owner.
        owner_org_id: "local".into(),
        name: rec.name,
        description: rec.description,
        public: false,
        region: None,
        provider: None,
        platform: s.platform.clone(),
        resources: VmResources {
            vcpu: rec.resources.vcpu,
            memory_mib: rec.resources.memory_mib,
            disk_mib: rec.resources.disk_mib,
        },
        network: VmNetwork::default(),
        pool_id: s.pool_id.clone(),
        vm_id: rec.vm_id,
    }
}

pub async fn get_regions(State(s): State<AppState>) -> ApiResult<ListRegionsResponse> {
    Ok(Json(ListRegionsResponse {
        regions: vec![RegionEntry {
            provider: "blast".into(),
            region: "local".into(),
            endpoint: format!("http://localhost:{}", s.config.port),
            platform: s.platform.clone(),
            vcpu: s.total_resources.vcpu,
            memory_mib: s.total_resources.memory_mib,
            disk_mib: s.total_resources.disk_mib,
        }],
    }))
}

pub async fn post_fork(
    State(s): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ForkRequest>,
) -> ApiResult<Vm> {
    let vm_id = headers
        .get(VM_ID_HEADER)
        .map(|v| v.to_str().map(str::to_owned))
        .transpose()
        .map_err(|_| ApiError::bad_request(format!("{VM_ID_HEADER} is not valid text")))?;
    // The id names directories and object keys.
    if vm_id
        .as_deref()
        .is_some_and(|id| !crate::snapshot::is_safe_vm_id(id))
    {
        return Err(ApiError::bad_request(format!(
            "{VM_ID_HEADER} must be 1 to 128 letters, digits, underscores or hyphens"
        )));
    }
    handle_fork(&s, vm_id, req).await.map(Json)
}

/// Refuses fork options this worker cannot honour, rather than
/// silently ignoring them.
fn check_fork_supported(s: &AppState, req: &ForkRequest) -> Result<(), ApiError> {
    let non_empty = |v: &Option<serde_json::Value>| {
        v.as_ref().is_some_and(|v| match v {
            serde_json::Value::Null => false,
            serde_json::Value::Array(a) => !a.is_empty(),
            serde_json::Value::Object(o) => !o.is_empty(),
            _ => true,
        })
    };
    // Building a Dockerfile is the SDKs' job, so the refusal names them.
    if req.dockerfile.is_some() {
        return Err(ApiError::bad_request_field(
            "dockerfile",
            "fork(dockerfile=...) is not supported by the API: build the image with your SDK's \
             fork(dockerfile=...), or pass a prebuilt `image`.",
        ));
    }
    if req.source_org_id.is_some() || req.source_org_name.is_some() {
        return Err(ApiError::unsupported(
            None,
            "cross-organization sources are not supported",
        ));
    }
    if req.nestedvirt == Some(true) {
        return Err(ApiError::unsupported(
            Some("nested_virtualization"),
            "nestedvirt is not supported",
        ));
    }
    if req.public == Some(true) {
        return Err(ApiError::unsupported(
            Some("public_ingress"),
            "public VMs are not supported",
        ));
    }
    if non_empty(&req.ssh_public_keys) {
        return Err(ApiError::unsupported(
            None,
            "ssh_public_keys is not supported",
        ));
    }
    if req.disk == Some(false) {
        return Err(ApiError::unsupported(
            Some("diskless"),
            "diskless VMs are not supported",
        ));
    }
    if req.durable == Some(true) {
        return Err(ApiError::unsupported(
            Some("durability"),
            "durable VMs are not supported",
        ));
    }
    if non_empty(&req.policies) {
        return Err(ApiError::unsupported(
            Some("network_policy"),
            "network policies are not supported",
        ));
    }
    if req.resources.as_ref().is_some_and(|r| non_empty(&r.vgpu)) {
        return Err(ApiError::unsupported(
            Some("platform"),
            "GPUs are not supported",
        ));
    }
    if req
        .layers
        .as_ref()
        .is_some_and(|l| !l.iter().any(|layer| layer == "memory"))
    {
        return Err(ApiError::unsupported(
            Some("filesystem_only_fork"),
            "filesystem-only forks are not supported; omit `layers`",
        ));
    }
    if let Some(platforms) = req.platforms.as_ref().filter(|p| !p.is_empty()) {
        if !platforms.contains(&s.platform) {
            return Err(ApiError::unsupported(
                Some("platform"),
                format!("this worker serves only platform {}", s.platform),
            ));
        }
    }
    Ok(())
}

/// The shape a fork gets: what it asks for, else its source's, else the
/// default.
fn fork_resources(req: &ForkRequest, source: Option<&VmRecord>) -> Result<Resources, ApiError> {
    let explicit = req.resources.as_ref().map(|r| Resources {
        vcpu: r.vcpu.unwrap_or(DEFAULT_RESOURCES.vcpu),
        memory_mib: r.memory_mib.unwrap_or(DEFAULT_RESOURCES.memory_mib),
        disk_mib: r.disk_mib.unwrap_or(DEFAULT_RESOURCES.disk_mib),
    });
    // A VM-fork always inherits the source's actual shape: SmolVM rejects a
    // resize on checkpoint restore, and Docker and Hypeman silently ignore it
    // (a fork asking for 4 vcpu from a 1-vcpu source still runs on 1). Checked
    // here, once, against BLAST's own record of the source.
    match (explicit, source) {
        (Some(want), Some(src)) if want != src.resources => Err(ApiError::unsupported(
            Some("resize"),
            format!(
                "cannot resize on fork from a VM: source {} is {}vcpu/{}MiB/{}MiB disk, requested \
                 {}vcpu/{}MiB/{}MiB disk. Omit `resources` to inherit the source's shape, or fork \
                 from an image for a specific size.",
                src.vm_id,
                src.resources.vcpu,
                src.resources.memory_mib,
                src.resources.disk_mib,
                want.vcpu,
                want.memory_mib,
                want.disk_mib
            ),
        )),
        (Some(want), _) => Ok(want),
        (None, Some(src)) => Ok(src.resources.clone()),
        (None, None) => Ok(DEFAULT_RESOURCES),
    }
}

pub async fn handle_fork(
    s: &AppState,
    vm_id: Option<String>,
    req: ForkRequest,
) -> Result<Vm, ApiError> {
    let arrived = Instant::now();
    let _mutation = begin_mutation(s).await?;
    check_fork_supported(s, &req)?;
    if req.image.is_some() && (req.source_vm_id.is_some() || req.source_vm_name.is_some()) {
        return Err(ApiError::bad_request(
            "give either `image` or a source VM, not both",
        ));
    }
    let vm_id = s
        .control_plane
        .allocate_vm_id(vm_id)
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let _target_vm = lock_vm(s, &vm_id).await;
    if s.store.get(&vm_id).await.is_some() {
        return Err(ApiError::already_exists(
            "vm",
            format!("VM already exists: {vm_id}"),
        ));
    }

    let (source, source_guard) = if req.image.is_some() {
        (None, None)
    } else {
        let (src, guard) = lock_source(s, &req).await?;
        (Some(src), Some(guard))
    };

    let resources = fork_resources(&req, source.as_ref())?;

    let mut record = VmRecord::new(vm_id.clone(), VmState::Running, resources);
    record.name = req.name.clone();
    record.description = req.description.clone().filter(|d| !d.trim().is_empty());
    record.image = req
        .image
        .clone()
        .or_else(|| source.as_ref().and_then(|src| src.image.clone()));
    record.parent_vm_id = source.as_ref().map(|src| src.vm_id.clone());
    record.workdir = source.as_ref().and_then(|src| src.workdir.clone());

    // Local admission control: a fork must fit in the declared pool, waiting
    // for headroom up to the queueing window. No pool means unlimited.
    let queue_secs = req
        .queueing_timeout
        .unwrap_or_else(|| s.config.admission_queue_secs());
    let pool =
        crate::admission::acquire_for_fork(&s.store, &s.total_resources, queue_secs, &record)
            .await?;

    let forked = resolve_fork_handle(
        s,
        &req,
        source.as_ref().zip(source_guard),
        arrived,
        &record.resources,
    )
    .await;
    let handle = match forked {
        Ok(h) => h,
        Err(e) => {
            crate::admission::release_failed_fork(&s.store, &vm_id, pool.as_ref()).await;
            return Err(e);
        }
    };
    record.handle = Some(handle.clone());
    match &source {
        None => record.workdir = image_workdir(s, &handle).await,
        // Being forked is activity of the source.
        Some(source) => {
            s.store
                .update(&source.vm_id, |r| r.last_active = Instant::now())
                .await;
            crate::lifecycle::publish(s, &source.vm_id).await;
        }
    }

    if pool.is_some() {
        // Promote the `Pending` placeholder reserved above.
        s.store
            .update(&vm_id, |r| {
                r.state = VmState::Running;
                r.handle = Some(handle.clone());
                r.workdir.clone_from(&record.workdir);
            })
            .await;
    } else {
        s.store.insert(record.clone()).await;
    }
    if let Err(error) = seed_sessions(s, &vm_id, source.as_ref()).await {
        let _ = s.backend.delete(&handle).await;
        s.store.remove(&vm_id).await;
        s.store.notify_capacity_freed();
        return Err(ApiError::internal(format!("persist sessions: {error}")));
    }
    let record = s.store.get(&vm_id).await.unwrap_or(record);
    persist_recoverable_fork(s, &record, &handle).await?;
    persist_vm_record(&s.records, &record)?;
    if let Err(error) = s.records.activity(Activity::operation(
        "fork",
        "completed",
        Some(record.vm_id.clone()),
    )) {
        tracing::warn!(vm_id = %record.vm_id, err = %error, "persist fork activity failed");
    }
    Ok(vm_object(s, record).await)
}

/// Gives a new VM its source's sessions and, whatever it was forked from,
/// the session its runs use when they name none.
async fn seed_sessions(
    s: &AppState,
    vm_id: &str,
    source: Option<&VmRecord>,
) -> std::io::Result<()> {
    if let Some(source) = source {
        s.store.inherit_sessions(vm_id, source).await?;
    }
    s.store
        .open_session(vm_id, Some(0), None, HashMap::new())
        .await
        .map(|_| ())
}

/// Where the image starts its commands: its working directory, or its user's
/// home when it declares none. A VM whose image cannot say runs them from
/// wherever its backend does.
async fn image_workdir(s: &AppState, handle: &VmHandle) -> Option<String> {
    let mut exec = Exec::internal(
        r#"d=$(pwd); [ "$d" = / ] && d=${HOME:-/}; printf '%s\n' "$d""#,
        Stdout::Capture,
    );
    exec.cwd = None;
    let out = s.backend.run(handle, exec).await.ok()?;
    let workdir = String::from_utf8(out.stdout.bytes).ok()?;
    let workdir = workdir.trim_end_matches('\n');
    (out.ended == Ended::Exited(0) && workdir.starts_with('/')).then(|| workdir.to_owned())
}

/// Persist the control-plane identity and backend handle before acknowledging
/// a fork. A daemon-backed hypervisor outlives this process, so the next BLAST
/// process needs the manifest to reconstruct its store and reattach.
async fn persist_recoverable_fork(
    s: &AppState,
    record: &VmRecord,
    handle: &VmHandle,
) -> Result<(), ApiError> {
    let recoverable = s
        .backend
        .can_recover(handle)
        .await
        .map_err(|error| ApiError::internal(format!("inspect new backend VM: {error:#}")))?;
    if !recoverable {
        return Ok(());
    }
    if let Err(error) = s
        .snapshots
        .write_manifest(&crate::lifecycle::manifest(record))
        .await
    {
        let _ = s.backend.delete(handle).await;
        s.store.remove(&record.vm_id).await;
        s.store.notify_capacity_freed();
        return Err(ApiError::internal(format!(
            "persist VM identity before acknowledging fork: {error:#}"
        )));
    }
    Ok(())
}

/// The fork source, locked so it holds still until its snapshot is pinned.
async fn lock_source(
    s: &AppState,
    req: &ForkRequest,
) -> Result<(VmRecord, OwnedMutexGuard<()>), ApiError> {
    let src_id = resolve_source_vm_id(s, req).await?;
    let guard = lock_vm(s, &src_id).await;
    let src = find_vm(s, &src_id).await.map_err(|e| {
        if e.status() == StatusCode::NOT_FOUND {
            source_not_found()
        } else {
            e
        }
    })?;
    // A checkpoint would carry the commands' processes without their runs.
    let running = s.runs.of_vm(&src_id).await;
    if running
        .iter()
        .any(|(_, run)| run.state == RunState::Running)
    {
        return Err(ApiError::unsupported(
            Some("live_run_fork"),
            "forking a VM while its runs are in flight is not supported; fork it once they end",
        ));
    }
    Ok((src, guard))
}

/// Resolve source by id first, then by name.
async fn resolve_source_vm_id(s: &AppState, req: &ForkRequest) -> Result<String, ApiError> {
    // A fork of a VM pulls no image: accepting credentials there would
    // answer for one that was never used.
    if req.image.is_none() && req.registry_auth.is_some() {
        return Err(ApiError::unsupported(
            None,
            "registry credentials apply only to a fork from an image",
        ));
    }
    if let Some(ref id) = req.source_vm_id {
        Ok(id.clone())
    } else if let Some(ref name) = req.source_vm_name {
        s.store
            .all()
            .await
            .into_iter()
            .find(|vm| vm.name.as_deref() == Some(name.as_str()))
            .map(|vm| vm.vm_id)
            .ok_or_else(source_not_found)
    } else {
        Err(ApiError::bad_request(
            "one of: image, source_vm_id, source_vm_name required",
        ))
    }
}

/// A missing fork source answers the same 404 whether named by id or by name.
fn source_not_found() -> ApiError {
    ApiError::not_found("vm")
}

/// Resolves a fork to a live backend handle: booting fresh from an OCI
/// image, or snapshotting the source and forking from that. The source's
/// lock is held only until its snapshot is pinned.
async fn resolve_fork_handle(
    s: &AppState,
    req: &ForkRequest,
    source: Option<(&VmRecord, OwnedMutexGuard<()>)>,
    arrived: Instant,
    resources: &Resources,
) -> Result<VmHandle, ApiError> {
    let Some((src, source_guard)) = source else {
        let image = req.image.as_deref().unwrap_or_default();
        return s
            .backend
            .fork_image(image, resources, req.registry_auth.as_ref())
            .await
            .map_err(|error| {
                tracing::warn!(image, err = %format!("{error:#}"), "image fork failed");
                ApiError::image_fork_failed(format!(
                    "failed to pull image '{}': {error:#}",
                    canonical_image(image)
                ))
            });
    };
    let snapshot = pin_source_snapshot(s, src, arrived).await?;
    // A portable snapshot holds the whole VM, so forking from the pinned
    // copy needs nothing more of the source.
    if s.backend.snapshots_are_portable() {
        drop(source_guard);
    }
    Ok(s.backend.fork_snapshot(snapshot.path(), resources).await?)
}

/// The image reference in full: a bare name is Docker Hub's, and a name with
/// no namespace is one of its official images.
pub(super) fn canonical_image(image: &str) -> String {
    let (first, rest) = image.split_once('/').unwrap_or(("", image));
    let names_a_registry = first.contains(['.', ':']) || first == "localhost";
    match (names_a_registry, first.is_empty()) {
        (true, _) => image.to_owned(),
        (false, true) => format!("docker.io/library/{rest}"),
        (false, false) => format!("docker.io/{image}"),
    }
}

/// A snapshot of `src` that stays readable until dropped, whatever happens
/// to the source meanwhile. The caller holds the source's lock.
async fn pin_source_snapshot(
    s: &AppState,
    src: &VmRecord,
    arrived: Instant,
) -> Result<Arc<TransientSnapshot>, ApiError> {
    if let Some(handle) = &src.handle {
        match shared_fork_checkpoint(s, &src.vm_id, handle, arrived).await {
            Ok(checkpoint) => return Ok(checkpoint),
            Err(error) if !crate::lifecycle::demote_if_stopped(s, src).await => return Err(error),
            Err(_) if s.store.get(&src.vm_id).await.is_none() => return Err(source_not_found()),
            Err(_) => {}
        }
    }
    // Suspended: fork from its snapshot, restored from storage if this
    // worker no longer has it.
    s.snapshots
        .ensure_local(&src.vm_id)
        .await
        .map_err(|e| storage_unavailable(&e))?;
    s.snapshots
        .check_backend_tag(&src.vm_id, s.backend.kind())
        .await?;
    let pinned = s
        .snapshots
        .pin_local(&src.vm_id)
        .await
        .map_err(|e| ApiError::internal(format!("pin source snapshot: {e:#}")))?;
    Ok(Arc::new(pinned))
}

/// Checkpoints a running source, or reuses a checkpoint another fork began
/// after this one arrived: that checkpoint already holds every change made
/// before this fork was requested.
async fn shared_fork_checkpoint(
    s: &AppState,
    vm_id: &str,
    handle: &VmHandle,
    arrived: Instant,
) -> Result<Arc<TransientSnapshot>, ApiError> {
    if let Some(checkpoint) = s.snapshots.fork_checkpoint_since(vm_id, arrived) {
        return Ok(checkpoint);
    }
    let started = Instant::now();
    let checkpoint = s.snapshots.transient_snapshot(vm_id).await?;
    s.backend.snapshot(handle, checkpoint.path()).await?;
    let checkpoint = Arc::new(checkpoint);
    s.snapshots
        .share_fork_checkpoint(vm_id, started, &checkpoint);
    Ok(checkpoint)
}

pub async fn delete_vm(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
) -> ApiResult<DeleteResponse> {
    let _mutation = begin_mutation(&s).await?;
    // Holding the VM's lock waits out a sync or a snapshot of it in flight
    // (a fork holds it only until its snapshot is pinned), and keeps new
    // runs from starting.
    let _vm = lock_vm(&s, &vm_id).await;
    find_vm(&s, &vm_id).await?;
    end_active_runs(&s, &vm_id).await?;
    crate::lifecycle::destroy_vm(&s, &vm_id)
        .await
        .map_err(deletion_error)?
        .ok_or_else(vm_not_found)?;
    Ok(Json(DeleteResponse { deleted: true }))
}

/// A deletion the control plane could not be asked for has changed nothing:
/// the VM is as it was, and the same request can be sent again.
pub(super) fn deletion_error(error: anyhow::Error) -> ApiError {
    match error.downcast_ref::<crate::worker::client::CallError>() {
        Some(crate::worker::client::CallError::Transient(_)) => ApiError::unavailable(format!(
            "the control plane did not take the deletion; retry it: {error:#}"
        )),
        _ => ApiError::from(error),
    }
}

/// How long a delete waits for its runs' final records to reach the control
/// plane, which refuses a run of a VM it has already deleted.
const RUN_RECORDS_FLUSH: Duration = Duration::from_secs(10);

pub(super) fn deleted_mid_run(vm_id: &str) -> String {
    format!("VM {vm_id} was deleted while this run was in flight; the exec did not finish")
}

/// Ends the runs in flight on a VM being deleted, and delivers their
/// records first.
async fn end_active_runs(s: &AppState, vm_id: &str) -> Result<(), ApiError> {
    super::runs::fail_active_runs(s, vm_id)
        .await
        .map_err(|e| ApiError::unavailable(format!("persist ended run: {e}")))?;
    if !s.records.flush_vm(vm_id, RUN_RECORDS_FLUSH).await {
        return Err(ApiError::unavailable(
            "the VM's run records have not reached the control plane yet; retry the deletion",
        ));
    }
    Ok(())
}

/// A VM is as it was forked: nothing about it can be changed in place.
pub async fn patch_vm(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
    Json(patch): Json<serde_json::Value>,
) -> ApiResult<Vm> {
    find_vm(&s, &vm_id).await?;
    Err(if patch.get("resources").is_some_and(|r| !r.is_null()) {
        ApiError::unsupported(Some("resize"), "resizing a VM is not supported")
    } else {
        ApiError::unsupported(Some("vm_update"), "updating a VM is not supported")
    })
}

pub async fn list_vms(State(s): State<AppState>) -> ApiResult<ListVmsResponse> {
    let mut records = s.store.all().await;
    records.sort_by_key(|r| r.created_at);
    let mut vms = Vec::with_capacity(records.len());
    for record in records {
        vms.push(vm_object(&s, record).await);
    }
    Ok(Json(ListVmsResponse { vms }))
}

pub async fn get_vm(State(s): State<AppState>, Path(vm_id): Path<String>) -> ApiResult<Vm> {
    let rec = peek_vm(&s, &vm_id).await?;
    Ok(Json(vm_object(&s, rec).await))
}

/// Validates a durable snapshot without making the VM locally visible.
pub async fn prepare_claim(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
    Query(claim): Query<ClaimToken>,
) -> Result<StatusCode, ApiError> {
    validate_claim_incarnation(&s, &claim)?;
    let _mutation = begin_mutation(&s).await?;
    let _vm = lock_vm(&s, &vm_id).await;
    if s.store.get(&vm_id).await.is_some() {
        if s.snapshots
            .has_claim(&vm_id, claim.generation)
            .await
            .map_err(|error| ApiError::unavailable(format!("read VM claim: {error:#}")))?
        {
            return Ok(StatusCode::NO_CONTENT);
        }
        return Err(ApiError::already_exists(
            "vm",
            format!("VM already exists on this worker: {vm_id}"),
        ));
    }
    if !s.snapshots.has_shared_storage() {
        return Err(ApiError::unavailable(
            "claim preparation requires shared snapshot storage",
        ));
    }
    s.snapshots
        .prepare_claim(
            &vm_id,
            claim.generation,
            &claim.snapshot_generation,
            s.backend.kind(),
        )
        .await
        .map_err(|e| ApiError::unavailable(format!("prepare VM claim: {e:#}")))?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn commit_claim(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
    Query(claim): Query<ClaimToken>,
) -> Result<StatusCode, ApiError> {
    validate_claim_incarnation(&s, &claim)?;
    let _mutation = begin_mutation(&s).await?;
    let _vm = lock_vm(&s, &vm_id).await;
    let manifest = s
        .snapshots
        .claim(&vm_id, claim.generation)
        .await
        .map_err(|error| ApiError::unavailable(format!("claim VM: {error:#}")))?;
    s.snapshots
        .select_generation(&vm_id, &claim.snapshot_generation)
        .await
        .map_err(|error| ApiError::unavailable(format!("record claimed snapshot: {error:#}")))?;
    let record = record_from_manifest(&s, manifest)
        .await?
        .ok_or_else(|| ApiError::unavailable("claimed VM has no snapshot"))?;
    s.store.insert(record.clone()).await;
    let portable = s
        .snapshots
        .snap_dir(&vm_id)
        .join(crate::snapshot::PORTABLE_STATE_FILE);
    if tokio::fs::try_exists(&portable)
        .await
        .map_err(|error| ApiError::unavailable(format!("read claimed VM state: {error}")))?
    {
        crate::store::restore_portable_vm_state(&portable, &vm_id, &s.store, &s.runs)
            .await
            .map_err(|error| {
                ApiError::unavailable(format!("restore claimed VM state: {error:#}"))
            })?;
    }
    persist_vm_record(&s.records, &record)?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn abandon_claim(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
    Query(claim): Query<AbandonClaimToken>,
) -> Result<StatusCode, ApiError> {
    if s.control_plane.incarnation() != Some(claim.incarnation.as_str()) {
        return Err(ApiError::unavailable("worker incarnation changed"));
    }
    let _mutation = begin_mutation(&s).await?;
    let _vm = lock_vm(&s, &vm_id).await;
    if !s
        .snapshots
        .has_claim(&vm_id, claim.generation)
        .await
        .map_err(|error| ApiError::unavailable(format!("read VM claim: {error:#}")))?
    {
        return Ok(StatusCode::NO_CONTENT);
    }
    let record = s.store.get(&vm_id).await;
    if let Some(handle) = record.as_ref().and_then(|record| record.handle.as_ref()) {
        s.backend
            .delete(handle)
            .await
            .map_err(|error| ApiError::unavailable(format!("delete abandoned VM: {error:#}")))?;
    }
    s.store.remove(&vm_id).await;
    crate::store::forget_portable_vm_state(&vm_id, &s.store, &s.runs)
        .await
        .map_err(|error| ApiError::unavailable(format!("forget abandoned VM state: {error:#}")))?;
    s.records
        .abandon_vm(&vm_id)
        .map_err(|error| ApiError::unavailable(format!("forget abandoned VM record: {error}")))?;
    s.snapshots
        .abandon_claim(&vm_id, claim.generation)
        .await
        .map_err(|error| ApiError::unavailable(format!("abandon VM claim: {error:#}")))?;
    s.store.notify_capacity_freed();
    Ok(StatusCode::NO_CONTENT)
}

#[derive(serde::Deserialize)]
pub struct AbandonClaimToken {
    generation: i64,
    incarnation: String,
}

#[derive(serde::Deserialize)]
pub struct ClaimToken {
    generation: i64,
    snapshot_generation: String,
    incarnation: String,
}

fn validate_claim_incarnation(s: &AppState, claim: &ClaimToken) -> Result<(), ApiError> {
    if s.control_plane.incarnation() == Some(claim.incarnation.as_str()) {
        Ok(())
    } else {
        Err(ApiError::unavailable("worker incarnation changed"))
    }
}

/// Returns a live backend handle, restoring the VM when needed.
pub async fn ensure_running_handle(s: &AppState, vm_id: &str) -> Result<VmHandle, ApiError> {
    let vm = find_vm(s, vm_id).await?;
    let original_state = vm.state.clone();
    let admit = || {
        crate::admission::acquire_for_resume(
            &s.store,
            &s.total_resources,
            s.config.admission_queue_secs(),
            vm_id,
            &vm.resources,
        )
    };
    match (&vm.state, vm.handle.clone()) {
        (VmState::Paused, Some(h)) => {
            admit().await?;
            match s.backend.unpause(&h).await {
                Ok(()) => {
                    // Only now that the backend call landed is it Running;
                    // until then the reservation keeps it `Pending`, out of
                    // the pressure loop's reach.
                    s.store
                        .update(vm_id, |r| {
                            r.state = VmState::Running;
                            r.last_active = std::time::Instant::now();
                            r.idle_since = None;
                        })
                        .await;
                    crate::lifecycle::publish(s, vm_id).await;
                    Ok(h)
                }
                Err(e) => {
                    s.store.update(vm_id, |r| r.state = original_state).await;
                    s.store.notify_capacity_freed();
                    Err(e.into())
                }
            }
        }
        (_, Some(h)) => Ok(h),
        (_, None) => {
            // Restored from storage when the local copy is gone.
            let snap = s
                .snapshots
                .ensure_local(vm_id)
                .await
                .map_err(|e| storage_unavailable(&e))?;
            // A Docker image, a SmolVM checkpoint, and a Hypeman snapshot are
            // mutually unreadable: fail before spending a reservation.
            s.snapshots
                .check_backend_tag(vm_id, s.backend.kind())
                .await?;
            admit().await?;
            match s.backend.resume(&snap, &vm.resources).await {
                Ok(h) => {
                    let mut resumed = vm.clone();
                    resumed.state = VmState::Running;
                    resumed.handle = Some(h.clone());
                    resumed.last_active = std::time::Instant::now();
                    resumed.idle_since = None;
                    if let Err(error) = s
                        .snapshots
                        .write_manifest(&crate::lifecycle::manifest(&resumed))
                        .await
                    {
                        // A recoverable backend survives this process. Do not
                        // acknowledge a handle that a restarted worker would
                        // forget, or it could restore the stale checkpoint
                        // alongside the still-running machine.
                        if let Err(delete_error) = s.backend.delete(&h).await {
                            tracing::warn!(
                                vm_id,
                                err = %delete_error,
                                "failed to remove resumed VM after manifest write failed"
                            );
                        }
                        s.store
                            .update(vm_id, |r| {
                                r.state = original_state;
                                r.handle = None;
                            })
                            .await;
                        s.store.notify_capacity_freed();
                        return Err(ApiError::internal(format!(
                            "record resumed backend VM: {error:#}"
                        )));
                    }
                    s.store
                        .update(vm_id, |r| {
                            r.state = VmState::Running;
                            r.handle = Some(h.clone());
                            r.last_active = std::time::Instant::now();
                            r.idle_since = None;
                        })
                        .await;
                    crate::lifecycle::publish(s, vm_id).await;
                    Ok(h)
                }
                Err(e) => {
                    s.store
                        .update(vm_id, |r| {
                            r.state = original_state;
                            r.handle = None;
                        })
                        .await;
                    s.store.notify_capacity_freed();
                    Err(e.into())
                }
            }
        }
    }
}

pub async fn get_live() -> StatusCode {
    StatusCode::OK
}

/// Prometheus text exposition (no client library dependency: this is a
/// handful of gauges). The one metric this exists for is
/// `blast_pool_utilization_ratio` -- the signal an autoscaler needs to scale
/// before new forks start failing. `{0,0,0}` (no pool configured) skips
/// ratio/total output for that dimension: an undefined ratio is worse than
/// an absent one.
pub async fn get_metrics(State(s): State<AppState>) -> (HeaderMap, String) {
    let total = &s.total_resources;
    let in_use = s.store.resources_in_use().await;
    let vms = s.store.all().await;

    let mut out = String::new();
    // Prometheus text format requires HELP/TYPE to precede that metric
    // family's samples, and both name the bare metric.
    let mut metric = |name: &str, help: &str, samples: &[(String, String)]| {
        let _ = write!(out, "# HELP {name} {help}\n# TYPE {name} gauge\n");
        for (labels, value) in samples {
            let _ = writeln!(out, "{name}{labels} {value}");
        }
    };

    // One family for all three dimensions: HELP/TYPE may appear only once
    // per metric name, with its samples contiguous.
    let mut utilization_ratios: Vec<(String, String)> = Vec::new();

    if total.vcpu > 0 {
        metric(
            "blast_pool_vcpu_total",
            "Total vcpu configured for this worker's pool.",
            &[(String::new(), total.vcpu.to_string())],
        );
        metric(
            "blast_pool_vcpu_used",
            "Vcpu currently reserved by VMs on this worker.",
            &[(String::new(), in_use.vcpu.to_string())],
        );
        utilization_ratios.push((
            "{dimension=\"vcpu\"}".to_string(),
            (f64::from(in_use.vcpu) / f64::from(total.vcpu)).to_string(),
        ));
    }
    if total.memory_mib > 0 {
        metric(
            "blast_pool_memory_mib_total",
            "Total memory (MiB) configured for this worker's pool.",
            &[(String::new(), total.memory_mib.to_string())],
        );
        metric(
            "blast_pool_memory_mib_used",
            "Memory (MiB) currently reserved by VMs on this worker.",
            &[(String::new(), in_use.memory_mib.to_string())],
        );
        utilization_ratios.push((
            "{dimension=\"memory\"}".to_string(),
            ratio(in_use.memory_mib, total.memory_mib),
        ));
    }
    if total.disk_mib > 0 {
        metric(
            "blast_pool_disk_mib_total",
            "Total disk (MiB) configured for this worker's pool.",
            &[(String::new(), total.disk_mib.to_string())],
        );
        metric(
            "blast_pool_disk_mib_used",
            "Disk (MiB) currently reserved by VMs on this worker.",
            &[(String::new(), in_use.disk_mib.to_string())],
        );
        utilization_ratios.push((
            "{dimension=\"disk\"}".to_string(),
            ratio(in_use.disk_mib, total.disk_mib),
        ));
    }
    if !utilization_ratios.is_empty() {
        metric(
            "blast_pool_utilization_ratio",
            "Fraction of the pool currently reserved, per dimension.",
            &utilization_ratios,
        );
    }

    let vm_counts: Vec<(String, String)> = [
        VmState::Pending,
        VmState::Running,
        VmState::Paused,
        VmState::Suspended,
    ]
    .into_iter()
    .map(|state| {
        let count = vms.iter().filter(|v| v.state == state).count();
        let label = match state {
            VmState::Pending => "pending",
            VmState::Running => "running",
            VmState::Paused => "paused",
            VmState::Suspended => "suspended",
        };
        (format!("{{state=\"{label}\"}}"), count.to_string())
    })
    .collect();
    metric(
        "blast_vms",
        "Number of VMs on this worker by internal lifecycle state.",
        &vm_counts,
    );

    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    (headers, out)
}

#[allow(clippy::cast_precision_loss)]
fn ratio(used: u64, total: u64) -> String {
    (used as f64 / total as f64).to_string()
}
