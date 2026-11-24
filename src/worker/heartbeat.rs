//! `PUT {endpoint}/api/v1/pools/{pool_id}/workers/{worker_id}` with a
//! `WorkerHeartbeat` body, every `heartbeat_interval_ms` the response asks for.

use std::{collections::HashSet, path::Path, sync::atomic::Ordering, time::Duration};

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tracing::{info, warn};

use crate::{api::handlers::AppState, backend::Resources, snapshot::s3::ObjectStorage};

use super::client::{CallError, ControlPlane};

const FIRST_RETRY: Duration = Duration::from_secs(1);
const MAX_RETRY: Duration = Duration::from_secs(30);
const MIN_INTERVAL: Duration = Duration::from_secs(1);
const COMPATIBILITY_MEMBERSHIP_LEASE_MS: u64 = 30_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    Active,
    Draining,
    Left,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerResources {
    pub vcpu: u32,
    pub memory_mib: u64,
    pub disk_mib: u64,
}

impl From<&Resources> for WorkerResources {
    fn from(r: &Resources) -> Self {
        Self {
            vcpu: r.vcpu,
            memory_mib: r.memory_mib,
            disk_mib: r.disk_mib,
        }
    }
}

/// The control plane's `WorkerHeartbeat`. VMs are not listed: they are
/// reported one by one through the records API (`super::records`).
#[derive(Debug, Serialize)]
pub struct WorkerHeartbeat {
    pub kind: &'static str,
    pub version: &'static str,
    pub platform: String,
    pub state: WorkerState,
    pub resources: WorkerResources,
    pub free: WorkerResources,
}

/// The control plane's `WorkerHeartbeatResponse`.
#[derive(Debug, Deserialize)]
pub struct WorkerHeartbeatResponse {
    pub worker_id: String,
    pub heartbeat_interval_ms: u64,
    #[serde(default = "compatibility_membership_lease_ms")]
    pub membership_lease_ms: u64,
    pub desired_state: WorkerState,
    pub tunnel_url: String,
    /// Where snapshots go when the worker has no `storage_dir` of its own.
    pub storage: ObjectStorage,
}

const fn compatibility_membership_lease_ms() -> u64 {
    COMPATIBILITY_MEMBERSHIP_LEASE_MS
}

/// Why the heartbeat loop stopped.
#[derive(Debug, PartialEq, Eq)]
pub enum Ended {
    /// The pool is being deleted: the control plane wants this worker `left`.
    PoolDeleted,
    /// The control plane refused this worker for good; see [`refuses_for_good`].
    Refused { status: StatusCode, body: String },
    /// The worker id has left the pool: its VMs are other workers' now, and
    /// this machine may only join again as a new worker.
    Retired,
    /// Heartbeats could not renew membership before another worker may adopt
    /// this worker's VMs.
    LeaseExpired,
}

impl Ended {
    /// How a heartbeat refused for good ends the worker: 410 says its id has
    /// left the pool.
    fn refused(status: StatusCode, body: String) -> Self {
        if status == StatusCode::GONE {
            Self::Retired
        } else {
            Self::Refused { status, body }
        }
    }
}

/// A heartbeat answered with this status will never be accepted. Temporary
/// control-plane conditions use 5xx, which the client retries.
pub const fn refuses_for_good(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::CONFLICT | StatusCode::GONE | StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
    )
}

const fn desired_draining(state: WorkerState) -> Option<bool> {
    match state {
        WorkerState::Active => Some(false),
        WorkerState::Draining => Some(true),
        WorkerState::Left => None,
    }
}

pub async fn build(app: &AppState, advertised: &Resources, state: WorkerState) -> WorkerHeartbeat {
    let in_use = app.store.resources_in_use().await;
    body(&app.platform, advertised, &in_use, state)
}

pub fn body(
    platform: &str,
    advertised: &Resources,
    in_use: &Resources,
    state: WorkerState,
) -> WorkerHeartbeat {
    WorkerHeartbeat {
        kind: "blast",
        version: env!("CARGO_PKG_VERSION"),
        platform: platform.to_owned(),
        state,
        resources: advertised.into(),
        free: WorkerResources {
            vcpu: advertised.vcpu.saturating_sub(in_use.vcpu),
            memory_mib: advertised.memory_mib.saturating_sub(in_use.memory_mib),
            disk_mib: advertised.disk_mib.saturating_sub(in_use.disk_mib),
        },
    }
}

/// The configured `[resources]` pool, or the host's own size when none is
/// configured (admission is then unlimited, but the control plane still
/// needs a size to place against).
pub fn advertised_resources(configured: &Resources, data_dir: &Path) -> Resources {
    if configured.vcpu > 0 || configured.memory_mib > 0 || configured.disk_mib > 0 {
        return configured.clone();
    }
    Resources {
        vcpu: std::thread::available_parallelism()
            .map_or(1, |n| u32::try_from(n.get()).unwrap_or(u32::MAX)),
        memory_mib: host_memory_mib(),
        disk_mib: disk_mib(data_dir),
    }
}

#[cfg(unix)]
fn host_memory_mib() -> u64 {
    // SAFETY: sysconf only reads system configuration.
    let (pages, page_size) = unsafe {
        (
            libc::sysconf(libc::_SC_PHYS_PAGES),
            libc::sysconf(libc::_SC_PAGESIZE),
        )
    };
    let bytes = u64::try_from(pages).unwrap_or(0) * u64::try_from(page_size).unwrap_or(0);
    bytes / (1024 * 1024)
}

#[cfg(windows)]
fn host_memory_mib() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = u32::try_from(std::mem::size_of::<MEMORYSTATUSEX>()).unwrap_or(u32::MAX);
    if unsafe { GlobalMemoryStatusEx(&raw mut status) } == 0 {
        return 0;
    }
    status.ullTotalPhys / (1024 * 1024)
}

#[cfg(not(any(unix, windows)))]
fn host_memory_mib() -> u64 {
    0
}

#[cfg(unix)]
fn disk_mib(path: &Path) -> u64 {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return 0;
    };
    // SAFETY: `stat` is a plain C struct statvfs fills in; `c_path` is a valid
    // NUL-terminated string for the duration of the call.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return 0;
    }
    #[allow(clippy::useless_conversion)] // the field widths differ by platform
    let bytes = u64::from(stat.f_blocks) * u64::from(stat.f_frsize);
    bytes / (1024 * 1024)
}

#[cfg(windows)]
fn disk_mib(path: &Path) -> u64 {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);
    let mut total = 0;
    if unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            std::ptr::null_mut(),
            &raw mut total,
            std::ptr::null_mut(),
        )
    } == 0
    {
        return 0;
    }
    total / (1024 * 1024)
}

#[cfg(not(any(unix, windows)))]
fn disk_mib(_path: &Path) -> u64 {
    0
}

pub struct Loop {
    pub control_plane: ControlPlane,
    pub app: AppState,
    pub advertised: Resources,
    pub tunnel_url_tx: watch::Sender<Option<String>>,
}

/// Heartbeats until aborted, the pool is deleted, or the control plane
/// refuses this worker: on the interval the control plane asks for, and with
/// capped exponential backoff while it is unreachable. Each answer refreshes
/// the object storage credentials and drains or un-drains the worker.
pub async fn run(l: Loop) -> Ended {
    let mut retry = FIRST_RETRY;
    let mut joined = false;
    let mut fence_at = None;
    loop {
        if fence_at.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
            return fence(&l).await;
        }
        let state = if l.app.draining.load(Ordering::SeqCst) {
            WorkerState::Draining
        } else {
            WorkerState::Active
        };
        let heartbeat = async {
            let body = build(&l.app, &l.advertised, state).await;
            l.control_plane.heartbeat(&body).await
        };
        let result = match fence_at {
            Some(deadline) => match tokio::time::timeout_at(deadline, heartbeat).await {
                Ok(result) => result,
                Err(_) => return fence(&l).await,
            },
            None => heartbeat.await,
        };
        if fence_at.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
            return fence(&l).await;
        }
        let wait = match result {
            Ok(resp) => {
                let interval = Duration::from_millis(resp.heartbeat_interval_ms).max(MIN_INTERVAL);
                let next_fence = tokio::time::Instant::now() + fence_after(&resp, interval);
                fence_at = Some(next_fence);
                let Some(draining) = desired_draining(resp.desired_state) else {
                    quiesce(&l).await;
                    return Ended::PoolDeleted;
                };
                if l.app.draining.swap(draining, Ordering::SeqCst) != draining {
                    info!(draining, "the control plane changed this worker's state");
                }
                l.app.snapshots.set_object_storage(resp.storage).await;
                let admission = tokio::time::timeout_at(next_fence, admit(&l, !joined)).await;
                let Ok(admission) = admission else {
                    return fence(&l).await;
                };
                if let Err(error) = admission {
                    if joined {
                        warn!(err = %format!("{error:#}"), "publishing the admitted worker incarnation failed");
                        return fence(&l).await;
                    }
                    warn!(err = %format!("{error:#}"), retry_in = ?retry, "admitting worker-local state failed");
                    let wait = retry;
                    retry = (retry * 2).min(MAX_RETRY);
                    if tokio::time::timeout_at(next_fence, tokio::time::sleep(wait))
                        .await
                        .is_err()
                    {
                        return fence(&l).await;
                    }
                    continue;
                }
                l.control_plane.mark_ready();
                if !joined {
                    info!(worker_id = %resp.worker_id, "joined pool");
                    joined = true;
                }
                retry = FIRST_RETRY;
                let url = l.control_plane.resolve_tunnel_url(&resp.tunnel_url);
                l.tunnel_url_tx.send_if_modified(|current| {
                    let changed = current.as_deref() != Some(url.as_str());
                    *current = Some(url);
                    changed
                });
                interval
            }
            Err(CallError::Rejected { status, body }) if refuses_for_good(status) => {
                quiesce(&l).await;
                return Ended::refused(status, body);
            }
            Err(e) => {
                warn!(err = %format!("{e:#}"), retry_in = ?retry, "heartbeat failed");
                let wait = retry;
                retry = (retry * 2).min(MAX_RETRY);
                wait
            }
        };
        if let Some(deadline) = fence_at {
            if tokio::time::timeout_at(deadline, tokio::time::sleep(wait))
                .await
                .is_err()
            {
                return fence(&l).await;
            }
        } else {
            tokio::time::sleep(wait).await;
        }
    }
}

async fn admit(l: &Loop, reconcile: bool) -> anyhow::Result<()> {
    let incarnation = l
        .control_plane
        .incarnation()
        .ok_or_else(|| anyhow::anyhow!("a local control plane answered a heartbeat"))?;
    super::publish_incarnation(&l.app.config.data_dir, incarnation)?;
    if reconcile {
        reconcile_local_ownership(l).await?;
    }
    Ok(())
}

async fn reconcile_local_ownership(l: &Loop) -> anyhow::Result<()> {
    let _gate = l.app.lifecycle_gate.write().await;
    let local_vms = l.app.store.all().await;
    let pending: HashSet<_> = l
        .app
        .snapshots
        .pending_local_discards()
        .await?
        .into_iter()
        .collect();
    let local_vm_ids: Vec<_> = local_vms
        .iter()
        .filter(|vm| !pending.contains(&vm.vm_id))
        .map(|vm| vm.vm_id.clone())
        .collect();
    let offered: HashSet<_> = local_vm_ids.iter().cloned().collect();
    let retained_vm_ids = l.control_plane.reconcile_vms(&local_vm_ids).await?;
    let retained: HashSet<_> = retained_vm_ids.iter().cloned().collect();
    if retained.len() != retained_vm_ids.len() {
        anyhow::bail!("control plane returned a VM more than once during reconciliation");
    }
    if !retained.is_subset(&offered) {
        anyhow::bail!("control plane retained a VM that was not offered for reconciliation");
    }
    let mut local_vms: std::collections::HashMap<_, _> = local_vms
        .into_iter()
        .map(|vm| (vm.vm_id.clone(), vm))
        .collect();
    let mut discards = pending;
    discards.extend(offered.difference(&retained).cloned());
    for vm_id in discards {
        l.app.snapshots.begin_local_discard(&vm_id).await?;
        let vm = local_vms.remove(&vm_id);
        if let Some(handle) = vm.as_ref().and_then(|vm| vm.handle.as_ref()) {
            l.app.backend.delete(handle).await?;
        }
        l.app.store.remove(&vm_id).await;
        crate::store::forget_portable_vm_state(&vm_id, &l.app.store, &l.app.runs).await?;
        l.app.records.abandon_vm(&vm_id)?;
        l.app.snapshots.discard_local_vm_state(&vm_id).await?;
        l.app.snapshots.finish_local_discard(&vm_id).await?;
        l.app.store.notify_capacity_freed();
        info!(%vm_id, "discarded VM no longer owned by this worker");
    }
    Ok(())
}

fn fence_after(response: &WorkerHeartbeatResponse, interval: Duration) -> Duration {
    let lease = Duration::from_millis(response.membership_lease_ms);
    lease
        .checked_sub(interval.saturating_mul(2))
        .filter(|remaining| !remaining.is_zero())
        .unwrap_or_else(|| lease / 2)
}

async fn fence(l: &Loop) -> Ended {
    quiesce(l).await;
    warn!("membership lease is no longer renewable; worker self-fenced");
    Ended::LeaseExpired
}

async fn quiesce(l: &Loop) {
    l.app.draining.store(true, Ordering::SeqCst);
    l.tunnel_url_tx.send_replace(None);
    let _mutations_finished = l.app.lifecycle_gate.write().await;
}
