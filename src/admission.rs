//! Unified pool admission control: the ONE place any `fork` (new VM) or
//! `run` (resume from Paused/Suspended) request acquires vcpu/memory/disk
//! against the configured pool, and the ONE wait/queue/fail-fast loop both
//! go through. Previously this was two near-identical copy-pasted loops
//! inline in `api::handlers`; consolidating them here is what makes it
//! possible to state (and keep true) a single set of correctness claims for
//! all pool acquisition, not one set per copy.
//!
//! Release (giving resources back) is the mirror image but stays out of
//! this module: pause/suspend/evict/delete are already centralized in
//! `lifecycle::transitions` and `api::handlers::handle_delete`, and each
//! calls `Store::notify_capacity_freed()` directly at the point the
//! resource is actually freed. Acquisition and release meet only through
//! `Store`'s atomic reserve/promote methods and its notify pair -- there is
//! no third, separate ledger to keep in sync.
//!
//! # Correctness
//!
//! - **No TOCTOU.** Every check-and-mutate (`Store::try_reserve`,
//!   `Store::try_promote_to_running`) happens under ONE write-lock critical
//!   section inside `Store`. Nothing here ever reads capacity in one lock
//!   acquisition and acts on it in a second, separate one -- the classic
//!   shape of a TOCTOU race. A failed attempt returns the in-use snapshot
//!   from that SAME critical section, purely for the error message.
//!
//! - **No deadlock.** A reservation is all-or-nothing across all three
//!   dimensions in a single atomic step (`try_reserve`/
//!   `try_promote_to_running` either commit the whole request or commit
//!   nothing). There is no "hold vcpu, then separately wait for memory"
//!   partial-acquisition pattern, so two callers can never each hold what
//!   the other needs -- the precondition for a wait-for cycle doesn't
//!   exist.
//!
//! - **No livelock.** Every wait is bounded by an absolute deadline
//!   (`queue_secs`, from `admission_queue_secs`). A request that can never
//!   be satisfied (too big for the pool outright) fails immediately without
//!   waiting at all; one that's merely unlucky always either succeeds or
//!   times out with a clear error -- it can never spin forever changing
//!   state without making progress.
//!
//! - **No thundering-herd overreaction.** `Store::pending_demand()` sums
//!   what parked requests actually need, per dimension, so
//!   `lifecycle::pressure_loop` reclaims only the dimension that's short
//!   (see its doc comment) rather than pausing/suspending/evicting on every
//!   dimension just because *something* is parked.
//!
//! - **Fairness is best-effort, not FIFO**, unlike arkerd's
//!   semaphore-backed run-admission (genuinely FIFO via
//!   `tokio::sync::Semaphore`). When capacity frees, every parked waiter
//!   wakes and races to re-acquire under the store's single lock; a waiter
//!   that loses the race simply loops back to waiting. This can't deadlock
//!   or livelock (the deadline still bounds it), but under sustained heavy
//!   contention a large, unlucky request could in principle be repeatedly
//!   out-raced by smaller ones until its deadline. BLAST's target scale
//!   (a single worker, not arkerd's fleet) doesn't currently justify a full
//!   FIFO queue for this; revisit if that changes.

use std::time::{Duration, Instant};

use crate::{backend::Resources, store::Store};

/// Safety-net poll interval in case a `capacity_freed` notification is ever
/// missed. The common case is a near-immediate wake from the pressure loop,
/// not this fallback.
const FALLBACK_POLL: Duration = Duration::from_millis(500);

/// The two shapes `Store` supports acquiring against the pool -- kept as an
/// enum (rather than two copies of `acquire`) so there's exactly one wait
/// loop for both.
enum Reservation<'a> {
    /// Insert a brand-new `Pending` record (`Store::try_reserve`).
    New { vm_id: &'a str, name: Option<String> },
    /// Reserve toward promoting an existing Paused/Suspended record back to
    /// `Running` (`Store::try_reserve_resume`) -- marks it `Pending` on
    /// success; the caller flips it to `Running` once its own backend call
    /// (unpause/resume) actually succeeds.
    Resume { vm_id: &'a str },
}

impl Reservation<'_> {
    async fn attempt(
        &self,
        store: &Store,
        requested: &Resources,
        total: &Resources,
    ) -> Result<(), Resources> {
        match self {
            Self::New { vm_id, name } => {
                store.try_reserve(vm_id, name.clone(), requested, total).await
            }
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
) -> anyhow::Result<Option<Resources>> {
    if total.vcpu == 0 && total.memory_mib == 0 && total.disk_mib == 0 {
        return Ok(None);
    }
    if requested.vcpu > total.vcpu
        || requested.memory_mib > total.memory_mib
        || requested.disk_mib > total.disk_mib
    {
        anyhow::bail!(
            "requested resources ({} vcpu, {} MiB memory, {} MiB disk) exceed this \
             worker's total pool ({} vcpu, {} MiB memory, {} MiB disk) -- this can \
             never fit, regardless of what's freed up",
            requested.vcpu,
            requested.memory_mib,
            requested.disk_mib,
            total.vcpu,
            total.memory_mib,
            total.disk_mib,
        );
    }
    if reservation.attempt(store, requested, total).await.is_ok() {
        return Ok(Some(total.clone()));
    }

    let _waiter = store.register_admission_waiter(requested);
    let deadline = Instant::now() + Duration::from_secs(queue_secs);
    loop {
        let now = Instant::now();
        if now >= deadline {
            let in_use = store.resources_in_use().await;
            anyhow::bail!(
                "timed out after {queue_secs}s waiting for pool capacity (in use: \
                 {} vcpu, {} MiB memory, {} MiB disk of {} vcpu, {} MiB memory, \
                 {} MiB disk total)",
                in_use.vcpu,
                in_use.memory_mib,
                in_use.disk_mib,
                total.vcpu,
                total.memory_mib,
                total.disk_mib,
            );
        }
        let wait = (deadline - now).min(FALLBACK_POLL);
        let _ = tokio::time::timeout(wait, store.wait_capacity_freed()).await;
        if reservation.attempt(store, requested, total).await.is_ok() {
            return Ok(Some(total.clone()));
        }
        // Still doesn't fit: nudge the pressure loop again in case it went
        // back to sleep after a reclaim pass that wasn't enough on its own.
        store.request_admission_pass();
    }
}

/// Admits a brand-new fork. See the module doc for the correctness argument
/// and `Reservation::New` for what "admits" means mechanically.
pub async fn acquire_for_fork(
    store: &Store,
    total: &Resources,
    queue_secs: u64,
    vm_id: &str,
    name: Option<String>,
    requested: &Resources,
) -> anyhow::Result<Option<Resources>> {
    acquire(store, total, queue_secs, Reservation::New { vm_id, name }, requested).await
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
) -> anyhow::Result<Option<Resources>> {
    acquire(store, total, queue_secs, Reservation::Resume { vm_id }, requested).await
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
