use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, RwLock};
use ulid::Ulid;

use crate::backend::{Resources, VmHandle};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VmState {
    /// A pool reservation has been made and the backend fork is in flight;
    /// no handle yet. Holds vcpu+memory+disk exactly like `Running`, so a
    /// concurrent fork can't observe stale headroom and overcommit the pool
    /// while this one is still pulling an image / booting.
    Pending,
    /// VM process is running; CPU and memory are held.
    Running,
    /// VM is paused: CPU freed, memory still held by the backend.
    Paused,
    /// VM is suspended: CPU and memory freed; snapshot is on disk.
    Suspended,
}

#[derive(Debug, Clone)]
pub struct VmRecord {
    pub vm_id: String,
    pub name: Option<String>,
    pub state: VmState,
    pub resources: Resources,
    pub snapshot_dir: Option<PathBuf>,
    /// Backend handle, `None` when paused or suspended.
    pub handle: Option<VmHandle>,
    /// Wall-clock time of last activity (run, resume, fork).
    pub last_active: Instant,
    /// Wall-clock time the VM entered a non-running state (for TTL tracking).
    pub idle_since: Option<Instant>,
    pub sessions: HashMap<String, SessionRecord>,
}

#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub session_id: String,
    pub session_idx: u32,
    pub cwd: String,
    pub env: HashMap<String, String>,
    pub state: SessionState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionState {
    Idle,
    Running,
}

/// Sum of what every currently-parked admission request (fork or resume) is
/// waiting on, broken out per dimension. This is what lets the pressure
/// loop reclaim ONLY the dimension actually short, instead of a blanket
/// "someone's waiting, so pause+suspend+evict something" -- see
/// `admission::acquire` (the one place this is registered) and
/// `lifecycle::pressure_loop` (the one place it's read).
#[derive(Default)]
struct PendingDemand {
    vcpu: AtomicU32,
    memory_mib: AtomicU64,
    disk_mib: AtomicU64,
}

#[derive(Clone)]
pub struct Store {
    inner: Arc<RwLock<HashMap<String, VmRecord>>>,
    pending_demand: Arc<PendingDemand>,
    /// A parked admission request pings this to make the pressure loop run
    /// an immediate out-of-cycle reclaim pass instead of waiting out its
    /// normal tick. Mirrors arkerd's `notify.notify_one()` waking its
    /// scanner (`arkerd-worker-linux/src/background/resources.rs`).
    admission_requested: Arc<Notify>,
    /// Every transition that releases pool resources (pause, suspend, evict,
    /// delete) notifies this so parked requests recheck immediately rather
    /// than sitting out a fixed poll interval.
    capacity_freed: Arc<Notify>,
}

impl Default for Store {
    fn default() -> Self {
        Self {
            inner: Arc::default(),
            pending_demand: Arc::default(),
            admission_requested: Arc::new(Notify::new()),
            capacity_freed: Arc::new(Notify::new()),
        }
    }
}

/// RAII handle for a parked admission request: adds its requested resources
/// to the pending-demand totals on creation, subtracts them on drop
/// regardless of how the wait ends (success, timeout, or the caller's
/// future being dropped).
pub struct AdmissionWaiterGuard {
    demand: Arc<PendingDemand>,
    requested: Resources,
}

impl Drop for AdmissionWaiterGuard {
    fn drop(&mut self) {
        self.demand.vcpu.fetch_sub(self.requested.vcpu, Ordering::SeqCst);
        self.demand.memory_mib.fetch_sub(self.requested.memory_mib, Ordering::SeqCst);
        self.demand.disk_mib.fetch_sub(self.requested.disk_mib, Ordering::SeqCst);
    }
}

impl Store {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `requested` as pending demand for as long as the returned
    /// guard is held, and wakes the pressure loop to run an immediate
    /// reclaim pass. Only `admission::acquire` calls this -- it's the one
    /// place a request parks waiting for pool headroom.
    pub fn register_admission_waiter(&self, requested: &Resources) -> AdmissionWaiterGuard {
        self.pending_demand.vcpu.fetch_add(requested.vcpu, Ordering::SeqCst);
        self.pending_demand.memory_mib.fetch_add(requested.memory_mib, Ordering::SeqCst);
        self.pending_demand.disk_mib.fetch_add(requested.disk_mib, Ordering::SeqCst);
        self.admission_requested.notify_one();
        AdmissionWaiterGuard { demand: self.pending_demand.clone(), requested: requested.clone() }
    }

    /// Nudges the pressure loop again without registering new demand --
    /// used when a parked request rechecks and still doesn't fit, so a
    /// long-running pile-up keeps driving reclaim rather than only pinging
    /// once at the start of the wait.
    pub fn request_admission_pass(&self) {
        self.admission_requested.notify_one();
    }

    /// Sum of every currently-parked admission request's demand, per
    /// dimension. A plain atomics read -- no lock needed.
    pub fn pending_demand(&self) -> Resources {
        Resources {
            vcpu: self.pending_demand.vcpu.load(Ordering::SeqCst),
            memory_mib: self.pending_demand.memory_mib.load(Ordering::SeqCst),
            disk_mib: self.pending_demand.disk_mib.load(Ordering::SeqCst),
        }
    }

    /// Waits for the pressure loop to be pinged by a parked request.
    pub async fn wait_admission_requested(&self) {
        self.admission_requested.notified().await;
    }

    /// Waits for a signal that some transition just freed pool resources.
    /// Callers must still re-check capacity themselves on wakeup -- this
    /// only means "something changed, worth rechecking," not "it fits now."
    pub async fn wait_capacity_freed(&self) {
        self.capacity_freed.notified().await;
    }

    /// Called by every transition that releases pool resources (pause,
    /// suspend, evict) and by VM deletion.
    pub fn notify_capacity_freed(&self) {
        self.capacity_freed.notify_waiters();
    }

    pub async fn insert(&self, record: VmRecord) {
        self.inner.write().await.insert(record.vm_id.clone(), record);
    }

    pub async fn get(&self, vm_id: &str) -> Option<VmRecord> {
        self.inner.read().await.get(vm_id).cloned()
    }

    pub async fn update<F>(&self, vm_id: &str, f: F) -> bool
    where
        F: FnOnce(&mut VmRecord),
    {
        let mut map = self.inner.write().await;
        map.get_mut(vm_id).is_some_and(|rec| { f(rec); true })
    }

    pub async fn remove(&self, vm_id: &str) -> Option<VmRecord> {
        self.inner.write().await.remove(vm_id)
    }

    pub async fn all(&self) -> Vec<VmRecord> {
        self.inner.read().await.values().cloned().collect()
    }

    /// A suspended VM's checkpoint holds MORE disk than the VM's own
    /// declared `disk_mib`: suspend is pause + snapshot + kill, and the
    /// snapshot's memory dump alone writes roughly `memory_mib` worth of
    /// bytes to disk (confirmed directly: Hypeman's standby snapshot failed
    /// with a real ENOSPC writing that memory file when the pool's disk
    /// accounting had only ever charged `disk_mib`). Crediting
    /// `disk_mib + memory_mib` here is what makes the pool's admission
    /// control -- and the disk-pressure eviction it drives -- actually
    /// bound real host disk usage instead of a number that quietly
    /// undercounts every suspended VM.
    const fn suspended_disk_footprint(resources: &Resources) -> u64 {
        resources.disk_mib + resources.memory_mib
    }

    /// Resources currently held across all VMs, mirroring the exact hold
    /// semantics documented on : Pending and Running hold
    /// vcpu+memory+disk, Paused releases vcpu but keeps memory+disk,
    /// Suspended releases vcpu and memory but keeps disk (the snapshot,
    /// which itself needs the VM's `memory_mib` in extra disk -- see
    /// `suspended_disk_footprint`), and an evicted VM (removed from the
    /// store) holds nothing.
    fn resources_in_use_locked(map: &HashMap<String, VmRecord>) -> crate::backend::Resources {
        let mut vcpu = 0u32;
        let mut memory_mib = 0u64;
        let mut disk_mib = 0u64;
        for vm in map.values() {
            match vm.state {
                VmState::Pending | VmState::Running => {
                    vcpu += vm.resources.vcpu;
                    memory_mib += vm.resources.memory_mib;
                    disk_mib += vm.resources.disk_mib;
                }
                VmState::Paused => {
                    memory_mib += vm.resources.memory_mib;
                    disk_mib += vm.resources.disk_mib;
                }
                VmState::Suspended => {
                    disk_mib += Self::suspended_disk_footprint(&vm.resources);
                }
            }
        }
        crate::backend::Resources { vcpu, memory_mib, disk_mib }
    }

    pub async fn resources_in_use(&self) -> crate::backend::Resources {
        Self::resources_in_use_locked(&*self.inner.read().await)
    }

    /// Atomically checks `requested` against `total` (accounting for
    /// everything already held) and, if it fits, inserts a `Pending`
    /// placeholder reserving those resources -- all under one write-lock
    /// critical section, so two concurrent callers can never both pass the
    /// check for more than `total` combined allows. On success the caller
    /// owns the reservation under `vm_id` and must either promote it (via
    /// `update`, flipping state to `Running` and setting `handle`) or
    /// release it (via `remove`) once the backend fork resolves.
    ///
    /// Returns `Err(in_use)` when it doesn't fit right now -- the caller
    /// decides whether that's fail-fast (requested alone exceeds `total`,
    /// so it can never fit) or queue-and-retry (fits in `total` in
    /// principle, just busy).
    pub async fn try_reserve(
        &self,
        vm_id: &str,
        name: Option<String>,
        requested: &Resources,
        total: &Resources,
    ) -> Result<(), crate::backend::Resources> {
        let mut map = self.inner.write().await;
        let in_use = Self::resources_in_use_locked(&map);
        let fits = in_use.vcpu + requested.vcpu <= total.vcpu
            && in_use.memory_mib + requested.memory_mib <= total.memory_mib
            && in_use.disk_mib + requested.disk_mib <= total.disk_mib;
        if !fits {
            return Err(in_use);
        }
        map.insert(
            vm_id.to_string(),
            VmRecord {
                vm_id: vm_id.to_string(),
                name,
                state: VmState::Pending,
                resources: requested.clone(),
                snapshot_dir: None,
                handle: None,
                last_active: Instant::now(),
                idle_since: None,
                sessions: HashMap::new(),
            },
        );
        drop(map);
        Ok(())
    }

    /// Atomically checks whether promoting an EXISTING record to `Running`
    /// fits within `total`, then flips its state on success -- the resume
    /// counterpart to `try_reserve` (new VM) for the other admission-
    /// controlled moment: a `run()` call whose VM is Paused or Suspended
    /// re-acquires real vcpu (unpause) or vcpu+memory (resume), and must
    /// queue against the SAME pool rather than bypass it.
    ///
    /// On success, marks the record `Pending` (NOT `Running`) -- the actual
    /// backend `unpause`/`resume` call hasn't happened yet, and every
    /// pressure-loop reclaim function filters on an EXACT state match
    /// (`Running`/`Paused`/`Suspended`), so `Pending` stays invisible to all
    /// of them, same as a brand-new fork's placeholder. Setting it to
    /// `Running` here instead was a real bug: the pressure loop could pick
    /// this record for an eager pause while the caller's own `unpause()`
    /// was still in flight, racing two conflicting backend calls on the
    /// same handle (observed as Docker's "container is already paused").
    /// The caller must flip the record to `Running` itself once its backend
    /// call actually succeeds (see `api::handlers::ensure_running_handle`).
    ///
    /// Reads `vm_id`'s CURRENT state under the lock (not a value the caller
    /// captured earlier) to stay correct if something else -- the pressure
    /// loop, a concurrent request -- changed it since the caller last
    /// looked. A VM that's vanished (concurrently deleted) is treated as
    /// nothing to reserve; the caller's subsequent backend call surfaces
    /// "not found" on its own.
    pub async fn try_reserve_resume(
        &self,
        vm_id: &str,
        total: &Resources,
    ) -> Result<(), crate::backend::Resources> {
        let mut map = self.inner.write().await;
        let Some(rec) = map.get(vm_id) else { return Ok(()) };
        let full = rec.resources.clone();
        let held_now = match rec.state {
            VmState::Paused => {
                Resources { vcpu: 0, memory_mib: full.memory_mib, disk_mib: full.disk_mib }
            }
            VmState::Suspended => {
                Resources { vcpu: 0, memory_mib: 0, disk_mib: Self::suspended_disk_footprint(&full) }
            }
            VmState::Pending | VmState::Running => full.clone(),
        };
        let in_use = Self::resources_in_use_locked(&map);
        let fits = in_use.vcpu - held_now.vcpu + full.vcpu <= total.vcpu
            && in_use.memory_mib - held_now.memory_mib + full.memory_mib <= total.memory_mib
            && in_use.disk_mib - held_now.disk_mib + full.disk_mib <= total.disk_mib;
        if !fits {
            return Err(in_use);
        }
        map.get_mut(vm_id).expect("checked Some above").state = VmState::Pending;
        drop(map);
        Ok(())
    }
}

pub fn new_vm_id() -> String { format!("vm_{}", Ulid::new()) }
pub fn new_session_id() -> String { format!("sess_{}", Ulid::new()) }
pub fn new_run_id() -> String { format!("run_{}", Ulid::new()) }

// ── Runs ──────────────────────────────────────────────────────────────────

/// Lifecycle state for a Run, mirrors the control plane's `RunState`.
/// - `Running`: the command is in flight (either genuinely executing, or
///   the `time_to_background` sync window elapsed before it finished).
/// - `Completed`: the command ran to completion; `exit_code` conveys
///   success (0) or a non-zero program exit. A non-zero exit is still
///   `Completed`, the program ran.
/// - `Failed`: the system could not run or finish the command. `fail_reason`
///   explains why; this is distinct from `stderr` (the program's own output).
/// - `Cancelled`: the run was cancelled by the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunState {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl RunState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunRecord {
    pub state: RunState,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub stdout_encoding: Option<String>,
    pub stderr_encoding: Option<String>,
    pub exit_code: Option<i32>,
    pub fail_reason: Option<String>,
}

/// Poll store for run results, lets a caller that got back a `"running"`
/// state (because `time_to_background` elapsed before the command finished)
/// retrieve the eventual result via `GET /v1/vms/:vm_id/runs/:run_id`.
#[derive(Default, Clone)]
pub struct RunStore {
    inner: Arc<RwLock<HashMap<String, RunRecord>>>,
}

impl RunStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn insert(&self, run_id: String, record: RunRecord) {
        self.inner.write().await.insert(run_id, record);
    }

    pub async fn get(&self, run_id: &str) -> Option<RunRecord> {
        self.inner.read().await.get(run_id).cloned()
    }

    pub async fn update<F>(&self, run_id: &str, f: F) -> bool
    where
        F: FnOnce(&mut RunRecord),
    {
        let mut map = self.inner.write().await;
        map.get_mut(run_id).is_some_and(|rec| { f(rec); true })
    }
}
