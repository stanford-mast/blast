use std::time::Duration;

use tokio::time;
use tracing::{debug, info, warn};

use crate::{
    api::handlers::AppState,
    backend::Resources,
    config::LifecycleConfig,
    store::{VmRecord, VmState},
    worker::records::Activity,
};

mod transitions;

pub use transitions::demote_if_stopped;
pub use transitions::manifest;
pub use transitions::publish;
use transitions::{
    commit_generation, park_running_vm, resolve_pending, spawn_store_snapshot, suspend_vm,
    upload_with_lease,
};

/// How long after a failed attempt to store a suspended VM's snapshot the
/// next one is made.
const STORE_RETRY: Duration = Duration::from_secs(30);

/// Bundles the pool total and per-dimension pressure thresholds so
/// `pressure_loop` doesn't need one argument per field.
struct PressureThresholds {
    total: Resources,
    vcpu: f64,
    memory: f64,
}

/// VM state machine transitions (applied to idle VMs):
///
///   `Running` -(`pause_ttl` or vcpu pressure)-> `Paused`   [CPU freed, memory hot]
///   `Paused`  -(`suspend_ttl` or memory pressure)-> `Suspended`  [CPU+memory freed, snapshot on disk]
///
/// Backends without a real CPU-only pause move directly from Running to
/// Suspended. They never receive the Paused accounting discount.
///
/// A VM with an active session is never paused or suspended. Resource
/// accounting changes only after its backend transition succeeds.
///
/// Pressure is measured against the configured resource pool (`total`), not the
/// host OS. When `resources_in_use()` exceeds `total * (1 − threshold)` on a
/// given dimension, or pending demand exceeds its headroom, BLAST reclaims
/// that resource from the least recently used idle VM. Pressure is evaluated
/// per dimension, so CPU demand does not cause an unnecessary memory release.
pub fn spawn(cfg: &LifecycleConfig, app: &AppState) {
    let pause_ttl = Duration::from_secs(cfg.pause_ttl_secs);
    let suspend_ttl = Duration::from_secs(cfg.suspend_ttl_secs);

    {
        let app = app.clone();
        tokio::spawn(async move {
            ttl_loop(app, pause_ttl, suspend_ttl).await;
        });
    }

    let thresholds = PressureThresholds {
        total: app.total_resources.clone(),
        vcpu: cfg.vcpu_pressure_thresh,
        memory: cfg.memory_pressure_thresh,
    };
    let pressure_app = app.clone();
    tokio::spawn(async move {
        pressure_loop(pressure_app, thresholds).await;
    });

    let reconciliation_app = app.clone();
    tokio::spawn(async move {
        reconciliation_app.control_plane.ready().await;
        for vm in reconciliation_app.store.all().await {
            let pending = reconciliation_app
                .snapshots
                .pending_generation(&vm.vm_id)
                .await;
            match pending {
                Ok(Some(generation))
                    if generation.uploaded
                        && generation.kind == crate::snapshot::PendingKind::Suspended =>
                {
                    reconciliation_app
                        .store
                        .update(&vm.vm_id, |record| record.storing = true)
                        .await;
                    spawn_store_snapshot(&reconciliation_app, &vm.vm_id);
                }
                Ok(Some(generation)) if generation.uploaded => {
                    let _vm = crate::api::handlers::lock_vm(&reconciliation_app, &vm.vm_id).await;
                    if let Err(error) = resolve_pending(&reconciliation_app, &vm.vm_id).await {
                        warn!(vm_id = %vm.vm_id, err = %format!("{error:#}"), "reconciling dirty checkpoint failed");
                    }
                }
                Ok(Some(_)) => {
                    if let Err(error) = reconciliation_app
                        .snapshots
                        .abandon_generation(&vm.vm_id)
                        .await
                    {
                        warn!(vm_id = %vm.vm_id, err = %format!("{error:#}"), "abandoning partial snapshot generation failed");
                    }
                }
                _ => {}
            }
        }
    });

    if cfg.dirty_checkpoint_interval_secs > 0
        && app.backend.snapshots_are_portable()
        && app.snapshots.has_shared_storage()
    {
        let app = app.clone();
        let interval = Duration::from_secs(cfg.dirty_checkpoint_interval_secs);
        tokio::spawn(async move { dirty_checkpoint_loop(app, interval).await });
    }
}

async fn dirty_checkpoint_loop(app: AppState, interval: Duration) {
    app.control_plane.ready().await;
    let mut ticker = time::interval(interval);
    ticker.tick().await;
    loop {
        ticker.tick().await;
        if app.draining.load(std::sync::atomic::Ordering::SeqCst) {
            continue;
        }
        for vm in app.store.all().await {
            if vm.state != VmState::Running || vm.handle.is_none() {
                continue;
            }
            // Lock ordering matches API mutations: drain permit, then VM.
            // Otherwise a queued drain writer could deadlock with a request
            // that already holds its read permit while waiting for this VM.
            let gate = app.lifecycle_gate.clone().read_owned().await;
            let Some(vm_guard) = crate::api::handlers::try_lock_vm(&app, &vm.vm_id) else {
                // Active VMs are reconsidered on every tick. Skipping instead
                // of queueing avoids a stale checkpoint immediately after a
                // long mutation finishes.
                drop(gate);
                continue;
            };
            // Acquire the drain permit before spawning. A drain writer then
            // waits for every admitted checkpoint, including its upload.
            let checkpoint_app = app.clone();
            tokio::spawn(async move {
                let _gate = gate;
                let _vm = vm_guard;
                let Some(current) = checkpoint_app.store.get(&vm.vm_id).await else {
                    return;
                };
                if current.state != VmState::Running
                    || current.handle.is_none()
                    || current
                        .sessions
                        .values()
                        .any(|session| session.state == crate::store::SessionState::Running)
                    || checkpoint_app
                        .draining
                        .load(std::sync::atomic::Ordering::SeqCst)
                {
                    return;
                }
                checkpoint_running_vm(&checkpoint_app, &current).await;
            });
        }
    }
}

async fn checkpoint_running_vm(app: &AppState, vm: &VmRecord) {
    let dir = app.snapshots.checkpoint_dir(&vm.vm_id);
    let result = async {
        tokio::fs::create_dir_all(&dir).await?;
        app.backend
            .snapshot(vm.handle.as_ref().expect("checked handle"), &dir)
            .await?;
        let mut committed = vm.clone();
        committed.state = VmState::Suspended;
        committed.handle = None;
        app.snapshots
            .write_checkpoint_metadata(&dir, app.backend.kind(), &manifest(&committed))
            .await?;
        crate::store::write_portable_vm_state(
            &dir.join(crate::snapshot::PORTABLE_STATE_FILE),
            &vm.vm_id,
            &app.store,
            &app.runs,
        )
        .await?;
        if resolve_pending(app, &vm.vm_id).await? == Some(crate::snapshot::PendingKind::Suspended) {
            return Ok::<(), anyhow::Error>(());
        }
        let reservation = app.control_plane.reserve_snapshot(&vm.vm_id).await?;
        app.snapshots
            .set_pending_generation(
                &vm.vm_id,
                &crate::snapshot::PendingGeneration {
                    vm_generation: reservation.vm_generation,
                    snapshot_id: reservation.snapshot_id.clone(),
                    uploaded: false,
                    kind: crate::snapshot::PendingKind::DirtyCheckpoint,
                },
            )
            .await?;
        if let Err(error) = upload_with_lease(
            app,
            &vm.vm_id,
            &reservation,
            app.snapshots
                .persist_checkpoint_generation(&vm.vm_id, &dir, &reservation.snapshot_id),
        )
        .await
        {
            app.snapshots.abandon_generation(&vm.vm_id).await?;
            return Err(error);
        }
        app.snapshots.mark_generation_uploaded(&vm.vm_id).await?;
        commit_generation(app, &vm.vm_id, &reservation).await?;
        Ok::<(), anyhow::Error>(())
    }
    .await;
    let _ = tokio::fs::remove_dir_all(&dir).await;
    match result {
        Ok(()) => {
            info!(vm_id = %vm.vm_id, "stored dirty checkpoint");
            publish(app, &vm.vm_id).await;
        }
        Err(_) if demote_if_stopped(app, vm).await => {}
        Err(error) => {
            warn!(vm_id = %vm.vm_id, err = %format!("{error:#}"), "dirty checkpoint failed");
        }
    }
}

async fn ttl_loop(app: AppState, pause_ttl: Duration, suspend_ttl: Duration) {
    app.control_plane.ready().await;
    let mut ticker = time::interval(Duration::from_secs(1));
    let mut last_removal_retry = time::Instant::now();
    loop {
        ticker.tick().await;
        for vm in app.store.all().await {
            let Some(idle_since) = vm.idle_since else {
                continue;
            };
            if vm.state == VmState::Paused && idle_since.elapsed() >= suspend_ttl {
                let _gate = app.lifecycle_gate.read().await;
                let Some(_vm) = crate::api::handlers::try_lock_vm(&app, &vm.vm_id) else {
                    continue;
                };
                let Some(current) = app.store.get(&vm.vm_id).await else {
                    continue;
                };
                if current.state == VmState::Paused {
                    suspend_vm(&app, &current, "ttl").await;
                }
            }
        }

        for vm in app.store.all().await {
            if vm.state != VmState::Running {
                continue;
            }
            if vm.last_active.elapsed() < pause_ttl {
                continue;
            }
            let _gate = app.lifecycle_gate.read().await;
            let Some(_vm) = crate::api::handlers::try_lock_vm(&app, &vm.vm_id) else {
                continue;
            };
            let Some(current) = app.store.get(&vm.vm_id).await else {
                continue;
            };
            if current.state == VmState::Running {
                park_running_vm(&app, &current, "ttl").await;
            }
        }

        // A suspend whose snapshot could not be stored (storage unreachable,
        // or no object storage credentials yet) tries again, as does the
        // removal of a deleted VM's stored copy.
        if app.backend.snapshots_are_portable() && app.snapshots.can_store().await {
            for vm in app.store.all().await {
                let due = vm
                    .store_failed_at
                    .map_or(true, |at| at.elapsed() >= STORE_RETRY);
                if vm.state == VmState::Suspended && !vm.snapshot_stored && !vm.storing && due {
                    app.store.update(&vm.vm_id, |r| r.storing = true).await;
                    spawn_store_snapshot(&app, &vm.vm_id);
                }
            }
        }
        if last_removal_retry.elapsed() >= STORE_RETRY {
            last_removal_retry = time::Instant::now();
            app.snapshots.retry_removals().await;
        }
    }
}

async fn pressure_loop(app: AppState, thresholds: PressureThresholds) {
    app.control_plane.ready().await;
    let PressureThresholds {
        total,
        vcpu: vcpu_thresh,
        memory: mem_thresh,
    } = thresholds;
    // A 2s idle cadence plus an immediate wake on `admission_requested`: a
    // parked fork drives an out-of-cycle pass right away.
    let mut ticker = time::interval(Duration::from_secs(2));
    loop {
        tokio::select! {
            _ = ticker.tick() => {}
            () = app.store.wait_admission_requested() => {}
        }

        // Reclaim only the dimension that's actually short: a blanket "any
        // waiter pressures everything" rule needlessly suspends paused VMs
        // to fix memory that was never under pressure.
        let demand = app.store.pending_demand();

        if total.vcpu > 0 {
            let in_use = app.store.resources_in_use().await;
            let used_ratio = f64::from(in_use.vcpu) / f64::from(total.vcpu);
            let headroom = total.vcpu.saturating_sub(in_use.vcpu);
            let demand_pressure = demand.vcpu > 0 && headroom < demand.vcpu;
            if demand_pressure || used_ratio > 1.0 - vcpu_thresh {
                pause_one_running_eager(&app).await;
            }
        }

        if total.memory_mib > 0 {
            let in_use = app.store.resources_in_use().await;
            let used_ratio = f64::from(u32::try_from(in_use.memory_mib).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(total.memory_mib).unwrap_or(u32::MAX));
            let headroom = total.memory_mib.saturating_sub(in_use.memory_mib);
            let demand_pressure = demand.memory_mib > 0 && headroom < demand.memory_mib;
            if demand_pressure || used_ratio > 1.0 - mem_thresh {
                reclaim_memory(&app).await;
            }
        }

        if total.disk_mib > 0 {
            let in_use = app.store.resources_in_use().await;
            let headroom = total.disk_mib.saturating_sub(in_use.disk_mib);
            if demand.disk_mib > 0 && headroom < demand.disk_mib {
                reclaim_disk(&app).await;
            }
        }
    }
}

/// Eagerly pause the least-recently-active Running VM to reclaim vcpu under
/// pressure, without waiting for its normal `pause_ttl` to elapse.
async fn pause_one_running_eager(app: &AppState) {
    let mut running: Vec<_> = app
        .store
        .all()
        .await
        .into_iter()
        .filter(|v| v.state == VmState::Running)
        .collect();
    running.sort_by_key(|v| v.last_active);
    for vm in running {
        debug!(vm_id = %vm.vm_id, "eager pause (vcpu pressure)");
        let _gate = app.lifecycle_gate.read().await;
        let Some(_vm) = crate::api::handlers::try_lock_vm(app, &vm.vm_id) else {
            continue;
        };
        let Some(current) = app.store.get(&vm.vm_id).await else {
            continue;
        };
        if current.state == VmState::Running {
            park_running_vm(app, &current, "vcpu pressure").await;
            return;
        }
    }
}

async fn reclaim_memory(app: &AppState) {
    let mut candidates: Vec<_> = app
        .store
        .all()
        .await
        .into_iter()
        .filter(|v| matches!(v.state, VmState::Paused | VmState::Running))
        .collect();
    candidates.sort_by_key(|v| v.last_active);
    for vm in candidates {
        let _gate = app.lifecycle_gate.read().await;
        let Some(_vm) = crate::api::handlers::try_lock_vm(app, &vm.vm_id) else {
            continue;
        };
        let Some(mut current) = app.store.get(&vm.vm_id).await else {
            continue;
        };
        if current.state == VmState::Running {
            park_running_vm(app, &current, "memory pressure").await;
            let Some(updated) = app.store.get(&vm.vm_id).await else {
                continue;
            };
            current = updated;
        }
        if current.state == VmState::Paused {
            suspend_vm(app, &current, "memory pressure").await;
        }
        if current.state != VmState::Running {
            return;
        }
    }
}

async fn reclaim_disk(app: &AppState) {
    if !app.backend.snapshots_are_portable() || !app.snapshots.can_store().await {
        return;
    }
    let mut candidates = app.store.all().await;
    candidates.sort_by_key(|v| v.last_active);
    for vm in candidates {
        let _gate = app.lifecycle_gate.read().await;
        let Some(_vm) = crate::api::handlers::try_lock_vm(app, &vm.vm_id) else {
            continue;
        };
        let Some(mut current) = app.store.get(&vm.vm_id).await else {
            continue;
        };
        if current.state == VmState::Running {
            park_running_vm(app, &current, "disk pressure").await;
            let Some(updated) = app.store.get(&vm.vm_id).await else {
                continue;
            };
            current = updated;
        }
        if current.state == VmState::Paused {
            suspend_vm(app, &current, "disk pressure").await;
            return;
        }
        if current.state == VmState::Suspended && current.snapshot_dir.is_some() {
            if current.snapshot_stored {
                if app
                    .snapshots
                    .release_local_snapshot(&current.vm_id)
                    .await
                    .is_ok()
                {
                    app.store
                        .update(&current.vm_id, |record| record.snapshot_dir = None)
                        .await;
                    app.store.notify_capacity_freed();
                }
                return;
            }
            if !current.storing {
                app.store
                    .update(&current.vm_id, |record| record.storing = true)
                    .await;
                spawn_store_snapshot(app, &current.vm_id);
                return;
            }
        }
    }
}

/// Removes a VM: its record, its backend instance, and its snapshot, locally
/// and in storage. Returns `None` when no such VM exists.
pub async fn destroy_vm(app: &AppState, vm_id: &str) -> anyhow::Result<Option<VmRecord>> {
    if app.store.get(vm_id).await.is_none() {
        return Ok(None);
    }
    app.records
        .vm_deleted(vm_id)
        .await
        .map_err(anyhow::Error::new)?;
    crate::store::forget_portable_vm_state(vm_id, &app.store, &app.runs).await?;
    app.snapshots.forget(vm_id).await?;
    let Some(rec) = app.store.remove(vm_id).await else {
        return Ok(None);
    };
    if let Some(handle) = &rec.handle {
        if let Err(e) = app.backend.delete(handle).await {
            warn!(vm_id, err = %e, "backend delete failed");
        }
    }
    if let Err(error) = app.records.activity(Activity::operation(
        "delete",
        "completed",
        Some(vm_id.to_owned()),
    )) {
        warn!(vm_id, err = %error, "persist delete activity failed");
    }
    app.store.notify_capacity_freed();
    if rec.snapshot_dir.is_some() || rec.snapshot_stored {
        let snapshots = app.snapshots.clone();
        let vm_id = vm_id.to_owned();
        tokio::spawn(async move { snapshots.remove_stored(&vm_id).await });
    }
    Ok(Some(rec))
}

/// Puts every VM in a stored snapshot, so it outlives this process: this
/// worker serves it again after a restart, and any worker of the pool can
/// take it over. In-flight runs and copies finish first (up to `timeout`),
/// then each VM is paused and suspended, and its copy waited for (up to
/// `timeout` again).
pub async fn hand_back(
    app: &AppState,
    timeout: Duration,
    require_transferable: bool,
) -> anyhow::Result<()> {
    if !drain(app, timeout).await {
        anyhow::bail!("timed out waiting for VM work to quiesce");
    }
    for vm in app.store.all().await {
        if vm.state == VmState::Running {
            park_running_vm(app, &vm, "worker stopping").await;
        }
        if let Some(vm) = app.store.get(&vm.vm_id).await {
            if vm.state == VmState::Paused {
                suspend_vm(app, &vm, "worker stopping").await;
            }
        }
    }
    if !drain(app, timeout).await {
        anyhow::bail!("timed out waiting for snapshot copies to finish");
    }
    let mut failures = Vec::new();
    for vm in app.store.all().await {
        if vm.state != VmState::Suspended {
            warn!(vm_id = %vm.vm_id, state = ?vm.state, "VM could not be suspended; it does not outlive this process");
            failures.push(format!("{} is {:?}", vm.vm_id, vm.state));
        } else if require_transferable && !app.backend.snapshots_are_portable() {
            warn!(vm_id = %vm.vm_id, "VM is tied to this backend and prevents the worker from leaving");
            failures.push(format!("{} is not portable", vm.vm_id));
        } else if require_transferable && !app.snapshots.has_shared_storage() {
            warn!(vm_id = %vm.vm_id, "VM's snapshot is not in shared storage and prevents the worker from leaving");
            failures.push(format!("{} has no shared snapshot storage", vm.vm_id));
        } else if app.backend.snapshots_are_portable() && !vm.snapshot_stored {
            warn!(vm_id = %vm.vm_id, "VM's snapshot is not stored; only this machine has it");
            failures.push(format!("{} has no durable snapshot", vm.vm_id));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        anyhow::bail!("VM hand-back incomplete: {}", failures.join(", "))
    }
}

/// Lets in-flight runs and snapshot copies finish (up to `timeout`): a copy
/// cut short would leave the VM with no stored snapshot.
async fn drain(app: &AppState, timeout: Duration) -> bool {
    let deadline = time::Instant::now() + timeout;
    loop {
        let runs = app.runs.running_count().await;
        let records = app.store.all().await;
        let copies = records.iter().filter(|vm| vm.storing).count();
        let pending = records
            .iter()
            .filter(|vm| vm.state == VmState::Pending)
            .count();
        let busy = records
            .iter()
            .flat_map(|vm| vm.sessions.values())
            .filter(|session| session.state == crate::store::SessionState::Running)
            .count();
        if runs == 0 && copies == 0 && pending == 0 && busy == 0 {
            info!("drain: no runs or snapshot copies in flight");
            return true;
        }
        if time::Instant::now() >= deadline {
            warn!(runs, copies, pending, busy, "drain: timed out waiting");
            return false;
        }
        time::sleep(Duration::from_millis(200)).await;
    }
}
