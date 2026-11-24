//! Atomic resource admission for forks and resumes.

use std::time::{Duration, Instant};

use crate::{
    backend::Resources,
    store::{Store, VmRecord},
};

/// The pool cannot admit a request: it is larger than the pool outright, or
/// headroom did not free up within the queueing window.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct CapacityError(String);

/// The two shapes `Store` supports acquiring against the pool -- kept as an
/// enum (rather than two copies of `acquire`) so there's exactly one wait
/// loop for both.
enum Reservation<'a> {
    /// Insert a brand-new `Pending` record (`Store::try_reserve`).
    New { record: &'a VmRecord },
    /// Reserve toward promoting an existing Paused/Suspended record back to
    /// `Running` (`Store::try_reserve_resume`) -- marks it `Pending` on
    /// success; the caller flips it to `Running` once its own backend call
    /// (unpause/resume) actually succeeds.
    Resume { vm_id: &'a str },
}

impl Reservation<'_> {
    async fn attempt(&self, store: &Store, total: &Resources) -> Result<(), Resources> {
        match self {
            Self::New { record } => store.try_reserve(record, total).await,
            Self::Resume { vm_id } => store.try_reserve_resume(vm_id, total).await,
        }
    }
}

/// Reserves `requested` against `total`, waiting up to `queue_secs` for
/// headroom to free if it doesn't fit immediately, and failing fast if it
/// can never fit regardless of what frees up. `total` of `{0,0,0}` means no
/// pool is configured: returns `Ok(None)` immediately (unlimited, matching
/// `lifecycle::pressure_loop`'s own `total.x > 0` no-op guards).
async fn acquire(
    store: &Store,
    total: &Resources,
    queue_secs: u64,
    reservation: Reservation<'_>,
    requested: &Resources,
) -> Result<Option<Resources>, CapacityError> {
    if total.vcpu == 0 && total.memory_mib == 0 && total.disk_mib == 0 {
        return Ok(None);
    }
    let requested = &store.live_footprint(requested);
    if requested.vcpu > total.vcpu
        || requested.memory_mib > total.memory_mib
        || requested.disk_mib > total.disk_mib
    {
        return Err(CapacityError(format!(
            "requested resources ({} vcpu, {} MiB memory, {} MiB disk) exceed this \
             worker's total pool ({} vcpu, {} MiB memory, {} MiB disk) -- this can \
             never fit, regardless of what's freed up",
            requested.vcpu,
            requested.memory_mib,
            requested.disk_mib,
            total.vcpu,
            total.memory_mib,
            total.disk_mib,
        )));
    }
    if reservation.attempt(store, total).await.is_ok() {
        return Ok(Some(total.clone()));
    }

    let _waiter = store.register_admission_waiter(requested);
    let deadline = Instant::now() + Duration::from_secs(queue_secs);
    loop {
        if Instant::now() >= deadline {
            let in_use = store.resources_in_use().await;
            return Err(CapacityError(format!(
                "timed out after {queue_secs}s waiting for pool capacity (in use: \
                 {} vcpu, {} MiB memory, {} MiB disk of {} vcpu, {} MiB memory, \
                 {} MiB disk total)",
                in_use.vcpu,
                in_use.memory_mib,
                in_use.disk_mib,
                total.vcpu,
                total.memory_mib,
                total.disk_mib,
            )));
        }
        let changed = store.capacity_freed();
        tokio::pin!(changed);
        changed.as_mut().enable();
        if reservation.attempt(store, total).await.is_ok() {
            return Ok(Some(total.clone()));
        }
        store.request_admission_pass();
        let _ = tokio::time::timeout_at(deadline.into(), changed).await;
    }
}

/// Admits a brand-new fork. See the module doc for the correctness argument
/// and `Reservation::New` for what "admits" means mechanically.
pub async fn acquire_for_fork(
    store: &Store,
    total: &Resources,
    queue_secs: u64,
    record: &VmRecord,
) -> Result<Option<Resources>, CapacityError> {
    acquire(
        store,
        total,
        queue_secs,
        Reservation::New { record },
        &record.resources,
    )
    .await
}

/// Admits a resume: Paused -> Running (re-acquires vcpu) or Suspended /
/// no-handle -> Running (re-acquires vcpu + memory). `requested` is the
/// VM's own full resource shape (`VmRecord::resources`), not a delta --
/// `Store::try_promote_to_running` computes the delta itself from the
/// record's current state.
pub async fn acquire_for_resume(
    store: &Store,
    total: &Resources,
    queue_secs: u64,
    vm_id: &str,
    requested: &Resources,
) -> Result<Option<Resources>, CapacityError> {
    acquire(
        store,
        total,
        queue_secs,
        Reservation::Resume { vm_id },
        requested,
    )
    .await
}

/// Releases a reservation made by `acquire_for_fork` that never turned into
/// a live backend handle (the fork itself failed after admission
/// succeeded). No-op if `pool` is `None` (no pool was configured, so
/// nothing was reserved).
pub async fn release_failed_fork(store: &Store, vm_id: &str, pool: Option<&Resources>) {
    if pool.is_none() {
        return;
    }
    store.remove(vm_id).await;
    store.notify_capacity_freed();
}
