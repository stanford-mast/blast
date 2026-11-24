//! VM state transitions.

use std::{
    future::Future,
    time::{Duration, Instant, UNIX_EPOCH},
};

use tracing::{debug, info, warn};

use crate::{
    api::handlers::AppState,
    snapshot::{PendingGeneration, PendingKind, VmManifest},
    store::{SessionState, VmRecord, VmState},
    worker::records::Activity,
};

/// True if `vm` has no session with a run currently in flight, i.e. it is
/// safe to pause or suspend. Checked at the top of every transition below so
/// no caller can bypass it.
fn safe_to_transition(vm: &VmRecord) -> bool {
    !vm.sessions
        .values()
        .any(|s| s.state == SessionState::Running)
}

/// Reports the VM's current state to the control plane.
pub async fn publish(app: &AppState, vm_id: &str) {
    if let Some(vm) = app.store.get(vm_id).await {
        if let Err(error) = app.records.vm(&vm) {
            warn!(vm_id, err = %error, "persist VM record failed");
        }
    }
}

/// Running -> Paused: freeze CPU, keep memory resident.
pub async fn pause_vm(app: &AppState, vm: &VmRecord) {
    debug_assert!(app.backend.supports_pause());
    if !safe_to_transition(vm) {
        debug!(vm_id = %vm.vm_id, "skipping pause: session in flight");
        return;
    }
    let Some(handle) = vm.handle.clone() else {
        return;
    };
    match app.backend.pause(&handle).await {
        Ok(()) => {
            let now = Instant::now();
            app.store
                .update(&vm.vm_id, |r| {
                    r.state = VmState::Paused;
                    r.idle_since = Some(now);
                })
                .await;
            app.store.notify_capacity_freed();
            publish(app, &vm.vm_id).await;
            info!(vm_id = %vm.vm_id, "paused (CPU freed)");
        }
        Err(e) => warn!(vm_id = %vm.vm_id, err = %e, "pause failed"),
    }
}

/// Reclaim a running VM using the strongest state the backend can represent.
/// A backend without a real CPU-only pause goes straight to a durable
/// suspension, so its vCPU is released only after the VM process is gone.
pub async fn park_running_vm(app: &AppState, vm: &VmRecord, reason: &str) {
    if app.backend.supports_pause() {
        pause_vm(app, vm).await;
    } else {
        suspend_vm(app, vm, reason).await;
    }
}

/// Paused -> Suspended: write a snapshot to disk, destroy the VM process,
/// free CPU + memory, then copy the snapshot to durable storage.
pub async fn suspend_vm(app: &AppState, vm: &VmRecord, reason: &str) {
    if !safe_to_transition(vm) {
        debug!(vm_id = %vm.vm_id, "skipping suspend: session in flight");
        return;
    }
    let Some(handle) = vm.handle.as_ref() else {
        return;
    };
    let snap_dir = app.snapshots.snap_dir(&vm.vm_id);
    let suspended = match app.snapshots.invalidate_local_generation(&vm.vm_id).await {
        Ok(()) => app.backend.suspend(handle, &snap_dir).await,
        Err(error) => Err(error),
    };
    match suspended {
        Ok(()) => {
            let tag_written = match app
                .snapshots
                .write_backend_tag(&vm.vm_id, app.backend.kind())
                .await
            {
                Ok(()) => true,
                Err(e) => {
                    warn!(vm_id = %vm.vm_id, err = %e, "failed to record backend tag on suspend");
                    false
                }
            };
            let mut suspended = vm.clone();
            suspended.state = VmState::Suspended;
            suspended.handle = None;
            let manifest_written = match app.snapshots.write_manifest(&manifest(&suspended)).await {
                Ok(()) => true,
                Err(e) => {
                    warn!(vm_id = %vm.vm_id, err = %format!("{e:#}"), "failed to write the VM manifest on suspend");
                    false
                }
            };
            let state_written = crate::store::write_portable_vm_state(
                &snap_dir.join(crate::snapshot::PORTABLE_STATE_FILE),
                &vm.vm_id,
                &app.store,
                &app.runs,
            )
            .await
            .map_err(|error| {
                warn!(vm_id = %vm.vm_id, err = %format!("{error:#}"), "failed to write portable VM state on suspend");
            })
            .is_ok();
            let now = Instant::now();
            let store = tag_written
                && manifest_written
                && state_written
                && app.backend.snapshots_are_portable()
                && app.snapshots.can_store().await;
            app.store
                .update(&vm.vm_id, |r| {
                    r.state = VmState::Suspended;
                    r.handle = None;
                    r.snapshot_dir = Some(snap_dir);
                    r.idle_since = Some(now);
                    r.snapshot_stored = false;
                    r.storing = store;
                    r.store_failed_at = None;
                })
                .await;
            app.store.notify_capacity_freed();
            info!(vm_id = %vm.vm_id, %reason, "suspended");
            publish(app, &vm.vm_id).await;
            if let Err(error) = app.records.activity(Activity::operation(
                "vm_suspend",
                "completed",
                Some(vm.vm_id.clone()),
            )) {
                warn!(vm_id = %vm.vm_id, err = %error, "persist suspend activity failed");
            }
            if store {
                spawn_store_snapshot(app, &vm.vm_id);
            }
        }
        Err(_) if demote_if_stopped(app, vm).await => {}
        Err(e) => {
            // Back off before retrying: a VM whose backend can never
            // checkpoint it (e.g. smolvm cannot yet checkpoint a networked,
            // image-backed machine) would otherwise fail this call on every
            // 1s tick forever, spamming the log with an identical warning.
            let backoff_until = Instant::now();
            app.store
                .update(&vm.vm_id, |r| {
                    r.idle_since = Some(backoff_until);
                })
                .await;
            warn!(vm_id = %vm.vm_id, err = %e, "suspend failed, backing off");
            if let Err(error) = app.records.activity(Activity::operation(
                "vm_suspend",
                "failed",
                Some(vm.vm_id.clone()),
            )) {
                warn!(vm_id = %vm.vm_id, err = %error, "persist suspend failure activity failed");
            }
        }
    }
}

/// A VM whose machine stopped outside this worker's control resumes from
/// its last snapshot, or is deleted when it has none. Returns whether the
/// VM was demoted. The caller holds the VM's lock.
pub async fn demote_if_stopped(app: &AppState, vm: &VmRecord) -> bool {
    let Some(handle) = &vm.handle else {
        return false;
    };
    if !matches!(app.backend.is_stopped(handle).await, Ok(true)) {
        return false;
    }
    let vm_id = vm.vm_id.as_str();
    if let Err(error) = app.backend.delete(handle).await {
        warn!(vm_id, err = %format!("{error:#}"), "removing a stopped machine failed");
    }
    let last = match app.snapshots.last_snapshot(vm_id).await {
        Ok(last) => last,
        Err(error) => {
            warn!(vm_id, err = %format!("{error:#}"), "inspecting a stopped VM's snapshot failed");
            return false;
        }
    };
    let Some(last) = last else {
        warn!(
            vm_id,
            "VM's machine stopped and it has no snapshot; deleting it"
        );
        if let Err(error) = super::destroy_vm(app, vm_id).await {
            warn!(vm_id, err = %format!("{error:#}"), "deleting a lost VM failed");
        }
        return true;
    };
    let mut suspended = vm.clone();
    suspended.state = VmState::Suspended;
    suspended.handle = None;
    if let Err(error) = app.snapshots.write_manifest(&manifest(&suspended)).await {
        warn!(vm_id, err = %format!("{error:#}"), "recording a stopped VM as suspended failed");
    }
    let now = Instant::now();
    app.store
        .update(vm_id, |r| {
            r.state = VmState::Suspended;
            r.handle = None;
            r.snapshot_dir = last.local;
            r.snapshot_stored = last.stored;
            r.idle_since = Some(now);
            r.store_failed_at = None;
        })
        .await;
    app.store.notify_capacity_freed();
    warn!(
        vm_id,
        "VM's machine stopped; it resumes from its last snapshot"
    );
    publish(app, vm_id).await;
    true
}

pub fn manifest(vm: &VmRecord) -> VmManifest {
    VmManifest {
        vm_id: vm.vm_id.clone(),
        name: vm.name.clone(),
        description: vm.description.clone(),
        image: vm.image.clone(),
        parent_vm_id: vm.parent_vm_id.clone(),
        workdir: vm.workdir.clone(),
        created_at: vm
            .created_at
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        resources: vm.resources.clone(),
        backend_handle: vm.handle.as_ref().map(|handle| handle.id.clone()),
    }
}

/// Copies a suspended VM's snapshot to durable storage on its own task, so
/// a large upload never holds up the lifecycle loops. The caller has set the
/// record's `storing`, which keeps a second copy from starting meanwhile.
pub fn spawn_store_snapshot(app: &AppState, vm_id: &str) {
    let (app, vm_id) = (app.clone(), vm_id.to_owned());
    tokio::spawn(async move { store_snapshot(&app, &vm_id).await });
}

pub(super) async fn upload_with_lease<F>(
    app: &AppState,
    vm_id: &str,
    reservation: &crate::worker::client::SnapshotReservation,
    upload: F,
) -> anyhow::Result<()>
where
    F: Future<Output = anyhow::Result<()>>,
{
    if reservation.lease_renew_after_secs == 0 {
        return upload.await;
    }
    match app.control_plane.renew_snapshot(vm_id, reservation).await {
        Ok(()) => {}
        Err(crate::worker::client::CallError::Transient(error)) => {
            warn!(vm_id, err = %format!("{error:#}"), "snapshot lease renewal was delayed");
        }
        Err(error) => return Err(error.into()),
    }
    tokio::pin!(upload);
    let renew_after = Duration::from_secs(reservation.lease_renew_after_secs);
    loop {
        tokio::select! {
            result = &mut upload => return result,
            () = tokio::time::sleep(renew_after) => {
                match app.control_plane.renew_snapshot(vm_id, reservation).await {
                    Ok(()) => {}
                    Err(crate::worker::client::CallError::Transient(error)) => {
                        warn!(vm_id, err = %format!("{error:#}"), "snapshot lease renewal was delayed");
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }
    }
}

/// Snapshot persistence is serialized with every other operation on this VM.
/// Both the upload and local release use the snapshot directory, so allowing
/// resume, sync, or delete to overlap would race on that directory (and makes
/// removal fail outright on platforms such as Windows).
async fn store_snapshot(app: &AppState, vm_id: &str) {
    let _vm = crate::api::handlers::lock_vm(app, vm_id).await;
    let started = Instant::now();
    let stored = async {
        if resolve_pending(app, vm_id).await? == Some(PendingKind::Suspended) {
            return Ok::<(), anyhow::Error>(());
        }
        let reservation = app.control_plane.reserve_snapshot(vm_id).await?;
        app.snapshots
            .set_pending_generation(
                vm_id,
                &PendingGeneration {
                    vm_generation: reservation.vm_generation,
                    snapshot_id: reservation.snapshot_id.clone(),
                    uploaded: false,
                    kind: PendingKind::Suspended,
                },
            )
            .await?;
        if let Err(error) = upload_with_lease(
            app,
            vm_id,
            &reservation,
            app.snapshots
                .persist_generation(vm_id, &reservation.snapshot_id),
        )
        .await
        {
            app.snapshots.abandon_generation(vm_id).await?;
            return Err(error);
        }
        app.snapshots.mark_generation_uploaded(vm_id).await?;
        commit_generation(app, vm_id, &reservation).await?;
        Ok::<(), anyhow::Error>(())
    }
    .await;
    match stored {
        Ok(()) => {
            let released = app
                .snapshots
                .release_local_snapshot(vm_id)
                .await
                .map_err(|error| {
                    warn!(vm_id, err = %format!("{error:#}"), "durable snapshot stored but local disk was not released");
                    error
                })
                .is_ok();
            app.store
                .update(vm_id, |r| {
                    r.snapshot_stored = true;
                    r.storing = false;
                    if released {
                        r.snapshot_dir = None;
                    }
                })
                .await;
            if released {
                app.store.notify_capacity_freed();
            }
            info!(vm_id, took = ?started.elapsed(), "snapshot stored");
        }
        Err(e) => {
            app.store
                .update(vm_id, |r| {
                    r.snapshot_stored = false;
                    r.storing = false;
                    r.store_failed_at = Some(Instant::now());
                })
                .await;
            warn!(vm_id, err = %format!("{e:#}"), "storing the snapshot failed; will retry");
        }
    }
    if app.snapshots.removal_pending(vm_id).await {
        app.snapshots.remove_stored(vm_id).await;
    }
}

pub(super) async fn resolve_pending(
    app: &AppState,
    vm_id: &str,
) -> anyhow::Result<Option<PendingKind>> {
    let Some(pending) = app.snapshots.pending_generation(vm_id).await? else {
        return Ok(None);
    };
    if !pending.uploaded {
        app.snapshots.abandon_generation(vm_id).await?;
        return Ok(None);
    }
    let kind = pending.kind;
    let reservation = crate::worker::client::SnapshotReservation {
        vm_generation: pending.vm_generation,
        snapshot_id: pending.snapshot_id,
        lease_renew_after_secs: 0,
    };
    commit_generation(app, vm_id, &reservation).await?;
    Ok(Some(kind))
}

pub(super) async fn commit_generation(
    app: &AppState,
    vm_id: &str,
    reservation: &crate::worker::client::SnapshotReservation,
) -> anyhow::Result<()> {
    match app.control_plane.commit_snapshot(vm_id, reservation).await {
        Ok(()) => {
            app.snapshots
                .finish_generation(vm_id, &reservation.snapshot_id)
                .await
        }
        Err(crate::worker::client::CallError::Rejected { status, body }) => {
            app.snapshots.abandon_generation(vm_id).await?;
            Err(crate::worker::client::CallError::Rejected { status, body }.into())
        }
        Err(error) => Err(error.into()),
    }
}
