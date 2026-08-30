//! Canonical VM state-transition functions.
//!
//! Every place in `lifecycle` that moves a VM through
//! `Running -> Paused -> Suspended -> evicted` calls exactly one of the three
//! functions below, regardless of whether the trigger was TTL expiry or
//! resource pressure. All bookkeeping for a transition (backend call, store
//! mutation, logging) lives in one place per transition, see the commit that
//! introduced this module for the bug this fixes: `idle_since` was reset on
//! suspend in two separate inline copies of this logic, and only one of them
//! got fixed the first time around.
//!
//! Every transition also goes through [`safe_to_transition`] first: a VM with
//! any session mid-run must never be paused, suspended, or evicted out from
//! under that run, no matter how long the VM itself has looked idle.

use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::{
    backend::VmBackend,
    snapshot::SnapshotStore,
    store::{SessionState, Store, VmRecord, VmState},
};

/// True if `vm` has no session with a run currently in flight, i.e. it is
/// safe to pause, suspend, or evict. Checked at the top of every transition
/// below so no caller (TTL loop, pressure loop, or future reservation-based
/// triggers) can accidentally bypass it.
fn safe_to_transition(vm: &VmRecord) -> bool {
    !vm.sessions.values().any(|s| s.state == SessionState::Running)
}

/// Running -> Paused: freeze CPU, keep memory resident.
pub async fn pause_vm(store: &Store, backend: &Arc<dyn VmBackend>, vm: &VmRecord) {
    if !safe_to_transition(vm) {
        debug!(vm_id = %vm.vm_id, "skipping pause: session in flight");
        return;
    }
    let Some(handle) = vm.handle.clone() else { return };
    match backend.pause(&handle).await {
        Ok(()) => {
            let now = std::time::Instant::now();
            store
                .update(&vm.vm_id, |r| {
                    r.state = VmState::Paused;
                    r.idle_since = Some(now);
                })
                .await;
            store.notify_capacity_freed();
            info!(vm_id = %vm.vm_id, "paused (CPU freed)");
        }
        Err(e) => warn!(vm_id = %vm.vm_id, err = %e, "pause failed"),
    }
}

/// Paused -> Suspended: write a snapshot to disk, destroy the VM process,
/// free CPU + memory. Resets `idle_since` to the moment of suspension so
/// `evict_ttl` is measured from here, not from when the VM was paused.
pub async fn suspend_vm(
    store: &Store,
    backend: &Arc<dyn VmBackend>,
    snapshots: &Arc<SnapshotStore>,
    vm: &VmRecord,
    reason: &str,
) {
    if !safe_to_transition(vm) {
        debug!(vm_id = %vm.vm_id, "skipping suspend: session in flight");
        return;
    }
    let Some(handle) = vm.handle.as_ref() else { return };
    let snap_dir = snapshots.snap_dir(&vm.vm_id);
    match backend.suspend(handle, &snap_dir).await {
        Ok(()) => {
            // Best-effort: a failure here shouldn't undo an otherwise-good
            // suspend. `check_backend_marker` treats a missing marker as
            // "predates this check", not as a mismatch, so the worst case of
            // losing this write is the same as never having had it.
            if let Err(e) = snapshots.write_backend_marker(&vm.vm_id, backend.kind()).await {
                warn!(vm_id = %vm.vm_id, err = %e, "failed to record backend marker on suspend");
            }
            let now = std::time::Instant::now();
            store
                .update(&vm.vm_id, |r| {
                    r.state = VmState::Suspended;
                    r.handle = None;
                    r.snapshot_dir = Some(snap_dir);
                    r.idle_since = Some(now);
                })
                .await;
            store.notify_capacity_freed();
            info!(vm_id = %vm.vm_id, %reason, "suspended");
        }
        Err(e) => {
            // Back off before retrying: a VM whose backend can never
            // checkpoint it (e.g. smolvm cannot yet checkpoint a networked,
            // image-backed machine) would otherwise fail this call on every
            // 1s tick forever, spamming the log with an identical warning.
            let backoff_until = std::time::Instant::now();
            store
                .update(&vm.vm_id, |r| {
                    r.idle_since = Some(backoff_until);
                })
                .await;
            warn!(vm_id = %vm.vm_id, err = %e, "suspend failed, backing off");
        }
    }
}

/// Suspended -> removed: delete the store record and the on-disk snapshot.
///
/// This permanently destroys the last local copy of the VM's state, so it is
/// gated on `is_standalone`: a standalone worker (no `control_plane_endpoint`
/// configured) has no durable off-box copy to fall back on, so eviction is
/// skipped entirely there, suspension alone (snapshot stays on local disk)
/// is as far as the lifecycle goes.
pub async fn evict_vm(store: &Store, vm: &VmRecord, is_standalone: bool, reason: &str) {
    if !safe_to_transition(vm) {
        debug!(vm_id = %vm.vm_id, "skipping eviction: session in flight");
        return;
    }
    if is_standalone {
        info!(
            vm_id = %vm.vm_id,
            %reason,
            "skipping eviction: worker is standalone (no control plane), local snapshot is the only copy"
        );
        return;
    }
    store.remove(&vm.vm_id).await;
    if let Some(snap) = &vm.snapshot_dir {
        tokio::fs::remove_dir_all(snap).await.ok();
    }
    store.notify_capacity_freed();
    info!(vm_id = %vm.vm_id, %reason, "evicted");
}
