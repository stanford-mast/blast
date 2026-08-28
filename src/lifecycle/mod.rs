use std::{collections::HashMap, sync::Arc, time::Duration};

use tokio::time;
use tracing::{debug, warn};

use crate::{
    backend::{Resources, VmBackend},
    config::LifecycleConfig,
    snapshot::SnapshotStore,
    store::{SessionRecord, SessionState, Store, VmState},
};

mod transitions;

use transitions::{evict_vm, pause_vm, suspend_vm};

pub type UploadUrlFn = std::sync::Arc<
    dyn Fn(String, String)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<String>> + Send>>
        + Send
        + Sync,
>;

/// Bundles the pool total and per-dimension pressure thresholds so
/// `pressure_loop` doesn't need one argument per field.
struct PressureThresholds {
    total: Resources,
    vcpu: f64,
    memory: f64,
    disk: f64,
}

/// VM state machine transitions (applied to idle VMs):
///
///   `Running` -(`pause_ttl`)-> `Paused`   [CPU freed, memory hot]
///   `Paused`  -(`suspend_ttl` or memory pressure)-> `Suspended`  [CPU+memory freed, snapshot on disk]
///   `Suspended` -(`evict_ttl` or disk pressure)-> evicted      [record removed]
///
/// Every transition is a call into `transitions::{pause_vm,suspend_vm,evict_vm}`
///, that module is the single canonical place each transition's bookkeeping
/// (backend call, store mutation, logging) lives, and the single place the
/// concurrent-session guard is enforced (see `transitions::safe_to_transition`):
/// a VM with any session mid-run is never paused/suspended/evicted, regardless
/// of how idle the VM itself looks. `evict_vm` additionally skips entirely in
/// standalone mode (no control plane registered), since eviction destroys the
/// last local copy of a VM's snapshot and standalone workers have no durable
/// off-box copy to fall back on.
///
/// Pressure is measured against the configured resource pool (`total`), not the
/// host OS. When `resources_in_use()` exceeds `total * (1 − threshold)` on a
/// given dimension, OR `Store::pending_demand()` shows that dimension has
/// less headroom than parked requests need, BLAST eagerly pauses, suspends,
/// or evicts the LRU VM on THAT dimension to reclaim capacity (see
/// `admission::acquire`, the one place a fork or run-triggered resume parks
/// waiting for pool headroom). This mirrors arkerd's
/// `admission_waiters`-folded-into-`memory_pressure()` design
/// (`arkerd-worker-linux/src/background/resources.rs`): a pile-up of
/// requests that can't be admitted drives the same background
/// pause/suspend/evict machinery as ratio-threshold pressure, rather than a
/// second, disconnected polling loop -- but scoped per-dimension, so a
/// vcpu-only pile-up never triggers an unnecessary memory-driven suspend. A
/// parked request also pings `admission_requested` to wake this loop
/// immediately instead of waiting out its normal tick, and every successful
/// transition below calls `Store::notify_capacity_freed()` so parked
/// requests recheck right away.
///
/// Dirty-sync runs on running VMs only when an external upload URL is available.
pub fn spawn(
    cfg: &LifecycleConfig,
    store: Store,
    backend: &Arc<dyn VmBackend>,
    snapshots: Arc<SnapshotStore>,
    total_resources: &Resources,
    upload_url_fn: Option<UploadUrlFn>,
    is_standalone: bool,
) {
    let pause_ttl = Duration::from_secs(cfg.pause_ttl_secs);
    let suspend_ttl = Duration::from_secs(cfg.suspend_ttl_secs);
    let evict_ttl = Duration::from_secs(cfg.evict_ttl_secs);
    let dirty_sync_ttl = Duration::from_secs(cfg.dirty_sync_ttl_secs);
    let vcpu_thresh = cfg.vcpu_pressure_thresh;
    let mem_thresh = cfg.memory_pressure_thresh;
    let disk_thresh = cfg.disk_pressure_thresh;

    {
        let (store, backend, snapshots) = (store.clone(), backend.clone(), snapshots.clone());
        tokio::spawn(async move {
            ttl_loop(
                store, backend, snapshots, pause_ttl, suspend_ttl, evict_ttl, is_standalone,
            )
            .await;
        });
    }

    {
        let (store, backend, snapshots) = (store.clone(), backend.clone(), snapshots.clone());
        let thresholds = PressureThresholds {
            total: total_resources.clone(),
            vcpu: vcpu_thresh,
            memory: mem_thresh,
            disk: disk_thresh,
        };
        tokio::spawn(async move {
            pressure_loop(store, backend, snapshots, thresholds, is_standalone).await;
        });
    }

    if let Some(upload_fn) = upload_url_fn {
        let (store, backend, snapshots) = (store, backend.clone(), snapshots);
        tokio::spawn(async move {
            dirty_sync_loop(store, backend, snapshots, dirty_sync_ttl, upload_fn).await;
        });
    }
}

async fn ttl_loop(
    store: Store,
    backend: Arc<dyn VmBackend>,
    snapshots: Arc<SnapshotStore>,
    pause_ttl: Duration,
    suspend_ttl: Duration,
    evict_ttl: Duration,
    is_standalone: bool,
) {
    let mut ticker = time::interval(Duration::from_secs(1));
    loop {
        ticker.tick().await;
        for vm in store.all().await {
            let Some(idle_since) = vm.idle_since else { continue };
            let idle = idle_since.elapsed();

            match vm.state {
                VmState::Paused if idle >= suspend_ttl => {
                    suspend_vm(&store, &backend, &snapshots, &vm, "ttl").await;
                }

                VmState::Suspended if idle >= evict_ttl => {
                    evict_vm(&store, &vm, is_standalone, "ttl").await;
                }

                _ => {}
            }
        }

        for vm in store.all().await {
            if vm.state != VmState::Running { continue; }
            if vm.last_active.elapsed() < pause_ttl { continue; }
            pause_vm(&store, &backend, &vm).await;
        }
    }
}

async fn pressure_loop(
    store: Store,
    backend: Arc<dyn VmBackend>,
    snapshots: Arc<SnapshotStore>,
    thresholds: PressureThresholds,
    is_standalone: bool,
) {
    let PressureThresholds { total, vcpu: vcpu_thresh, memory: mem_thresh, disk: disk_thresh } =
        thresholds;
    // 2s idle cadence (down from a plain 5s tick) plus an immediate wake on
    // `admission_requested`: a parked fork drives an out-of-cycle pass right
    // away instead of waiting out the tick, same shape as arkerd's scanner
    // ticking faster (or waking on `notify`) under pressure.
    let mut ticker = time::interval(Duration::from_secs(2));
    loop {
        tokio::select! {
            _ = ticker.tick() => {}
            () = store.wait_admission_requested() => {}
        }

        // Per-dimension pending demand from parked admission requests (see
        // `admission::acquire`): reclaim the dimension that's ACTUALLY
        // short, not every dimension just because something is parked. A
        // blanket "any waiter pressures everything" heuristic was tried and
        // observed to needlessly suspend a merely-paused VM to "fix" memory
        // that was never actually under pressure -- precise per-dimension
        // headroom-vs-demand is what avoids that.
        let demand = store.pending_demand();

        if total.vcpu > 0 {
            let in_use = store.resources_in_use().await;
            let used_ratio = f64::from(in_use.vcpu) / f64::from(total.vcpu);
            let headroom = total.vcpu.saturating_sub(in_use.vcpu);
            let demand_pressure = demand.vcpu > 0 && headroom < demand.vcpu;
            if demand_pressure || used_ratio > 1.0 - vcpu_thresh {
                pause_one_running_eager(&store, &backend).await;
            }
        }

        if total.memory_mib > 0 {
            let in_use = store.resources_in_use().await;
            let used_ratio = f64::from(u32::try_from(in_use.memory_mib).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(total.memory_mib).unwrap_or(u32::MAX));
            let headroom = total.memory_mib.saturating_sub(in_use.memory_mib);
            let demand_pressure = demand.memory_mib > 0 && headroom < demand.memory_mib;
            if demand_pressure || used_ratio > 1.0 - mem_thresh {
                evict_one_paused(&store, &backend, &snapshots, "memory pressure").await;
            }
        }

        if total.disk_mib > 0 {
            let in_use = store.resources_in_use().await;
            let used_ratio = f64::from(u32::try_from(in_use.disk_mib).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(total.disk_mib).unwrap_or(u32::MAX));
            let headroom = total.disk_mib.saturating_sub(in_use.disk_mib);
            let demand_pressure = demand.disk_mib > 0 && headroom < demand.disk_mib;
            if demand_pressure || used_ratio > 1.0 - disk_thresh {
                evict_one_suspended(&store, "disk pressure", is_standalone).await;
            }
        }
    }
}

/// Eagerly pause the least-recently-active Running VM to reclaim vcpu under
/// pressure, without waiting for its normal `pause_ttl` to elapse.
async fn pause_one_running_eager(store: &Store, backend: &Arc<dyn VmBackend>) {
    let mut running: Vec<_> =
        store.all().await.into_iter().filter(|v| v.state == VmState::Running).collect();
    running.sort_by_key(|v| v.last_active);
    if let Some(vm) = running.first() {
        debug!(vm_id = %vm.vm_id, "eager pause (vcpu pressure)");
        pause_vm(store, backend, vm).await;
    }
}

/// Suspend the least-recently-active Paused VM to reclaim memory under pressure.
async fn evict_one_paused(
    store: &Store,
    backend: &Arc<dyn VmBackend>,
    snapshots: &Arc<SnapshotStore>,
    reason: &str,
) {
    let mut paused: Vec<_> = store
        .all()
        .await
        .into_iter()
        .filter(|v| v.state == VmState::Paused)
        .collect();
    paused.sort_by_key(|v| v.last_active);
    if let Some(vm) = paused.first() {
        suspend_vm(store, backend, snapshots, vm, reason).await;
    }
}

/// Evict the least-recently-active Suspended VM to reclaim disk under pressure.
async fn evict_one_suspended(store: &Store, reason: &str, is_standalone: bool) {
    let mut suspended: Vec<_> = store
        .all()
        .await
        .into_iter()
        .filter(|v| v.state == VmState::Suspended)
        .collect();
    suspended.sort_by_key(|v| v.last_active);
    if let Some(vm) = suspended.first() {
        evict_vm(store, vm, is_standalone, reason).await;
    }
}

async fn dirty_sync_loop(
    store: Store,
    backend: Arc<dyn VmBackend>,
    snapshots: Arc<SnapshotStore>,
    interval: Duration,
    upload_url_fn: UploadUrlFn,
) {
    // Invariant: `upload_url_fn` is only ever `Some` when a control plane is
    // configured (see src/main.rs, it's built inside the
    // `cfg.worker.control_plane_endpoint.is_some()` branch and this loop is
    // only spawned when `spawn()` receives `Some`). So dirty-sync is already
    // implicitly gated to non-standalone mode; no separate check needed here.
    let mut ticker = time::interval(interval);
    loop {
        ticker.tick().await;
        for vm in store.all().await {
            if vm.state != VmState::Running { continue; }
            let Some(handle) = vm.handle.clone() else { continue };
            let snap_dir = snapshots.snap_dir(&vm.vm_id);
            if let Err(e) = snapshots.ensure_dir(&vm.vm_id).await {
                warn!(vm_id = %vm.vm_id, err = %e, "dirty-sync: mkdir failed");
                continue;
            }
            // Same guard as api::handlers::resolve_fork_handle's
            // fork-from-running-source snapshot: mark the VM busy for the
            // duration so pause_vm/suspend_vm's TTL sweep of this same loop
            // iteration can't race a transition against the handle this
            // snapshot call is using.
            let guard_id = format!("dirty_sync_guard_{}", ulid::Ulid::new());
            store
                .update(&vm.vm_id, |r| {
                    r.sessions.insert(
                        guard_id.clone(),
                        SessionRecord {
                            session_id: guard_id.clone(),
                            session_idx: u32::MAX,
                            cwd: "/".into(),
                            env: HashMap::new(),
                            state: SessionState::Running,
                        },
                    );
                })
                .await;
            let snapshot_result = backend.snapshot(&handle, &snap_dir).await;
            store.update(&vm.vm_id, |r| { r.sessions.remove(&guard_id); }).await;
            if let Err(e) = snapshot_result {
                warn!(vm_id = %vm.vm_id, err = %e, "dirty-sync: snapshot failed");
                continue;
            }
            let upload_fn = upload_url_fn.clone();
            let vm_id = vm.vm_id.clone();
            if let Err(e) = snapshots
                .upload_via_presigned(&snap_dir, |hash| {
                    let f = upload_fn.clone();
                    let vid = vm_id.clone();
                    f(vid, hash)
                })
                .await
            {
                warn!(vm_id = %vm.vm_id, err = %e, "dirty-sync: upload failed");
            }
        }
    }
}
