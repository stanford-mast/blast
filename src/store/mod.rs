use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::{Instant, SystemTime},
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
    pub description: Option<String>,
    /// OCI image the VM (or its fork ancestor) booted from.
    pub image: Option<String>,
    /// The VM this one was forked from, if any.
    pub parent_vm_id: Option<String>,
    /// The image's working directory, where a run starts unless its session
    /// says otherwise. `None` for a VM made before this was recorded.
    pub workdir: Option<String>,
    pub created_at: SystemTime,
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
    /// Whether durable storage holds the current snapshot; meaningful only
    /// while suspended.
    pub snapshot_stored: bool,
    /// A copy to durable storage is in flight.
    pub storing: bool,
    /// When storing the snapshot last failed, so retries back off.
    pub store_failed_at: Option<Instant>,
}

impl VmRecord {
    pub fn new(vm_id: String, state: VmState, resources: Resources) -> Self {
        Self {
            vm_id,
            name: None,
            description: None,
            image: None,
            parent_vm_id: None,
            workdir: None,
            created_at: SystemTime::now(),
            state,
            resources,
            snapshot_dir: None,
            handle: None,
            last_active: Instant::now(),
            idle_since: None,
            sessions: HashMap::new(),
            snapshot_stored: false,
            storing: false,
            store_failed_at: None,
        }
    }

    /// The VM's sessions, without the internal busy sessions.
    pub fn client_sessions(&self) -> impl Iterator<Item = &SessionRecord> {
        self.sessions
            .values()
            .filter(|s| s.session_idx != BUSY_SESSION_IDX)
    }

    /// Whether a run or a sync is using the VM.
    pub fn in_use(&self) -> bool {
        self.sessions
            .values()
            .any(|s| s.session_idx == BUSY_SESSION_IDX)
    }

    /// Where a run of `session` starts, as the API reports it.
    pub fn session_cwd(&self, session: &SessionRecord) -> String {
        session
            .cwd
            .clone()
            .or_else(|| self.workdir.clone())
            .unwrap_or_else(|| "/".into())
    }
}

/// `session_idx` of the internal sessions `Store::mark_busy` inserts.
const BUSY_SESSION_IDX: u32 = u32::MAX;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: String,
    pub session_idx: u32,
    /// `None` starts runs in the image's working directory.
    pub cwd: Option<String>,
    pub env: HashMap<String, String>,
    /// `Running` only on the internal busy sessions.
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
    /// normal tick.
    admission_requested: Arc<Notify>,
    /// Every transition that releases pool resources (pause, suspend, evict,
    /// delete) notifies this so parked requests recheck immediately rather
    /// than sitting out a fixed poll interval.
    capacity_freed: Arc<Notify>,
    sessions: Arc<RwLock<SessionJournal>>,
    sessions_path: Option<Arc<PathBuf>>,
    /// Host memory a live VM holds beyond its guest RAM (the backend's VMM).
    vm_memory_overhead_mib: u64,
}

type SessionJournal = HashMap<String, HashMap<String, SessionRecord>>;

#[derive(Serialize, Deserialize)]
struct PortableVmState {
    sessions: HashMap<String, SessionRecord>,
    runs: HashMap<String, DurableRunRecord>,
}

impl Default for Store {
    fn default() -> Self {
        Self {
            inner: Arc::default(),
            pending_demand: Arc::default(),
            admission_requested: Arc::new(Notify::new()),
            capacity_freed: Arc::new(Notify::new()),
            sessions: Arc::default(),
            sessions_path: None,
            vm_memory_overhead_mib: 0,
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
        self.demand
            .vcpu
            .fetch_sub(self.requested.vcpu, Ordering::SeqCst);
        self.demand
            .memory_mib
            .fetch_sub(self.requested.memory_mib, Ordering::SeqCst);
        self.demand
            .disk_mib
            .fetch_sub(self.requested.disk_mib, Ordering::SeqCst);
    }
}

impl Store {
    pub fn durable(
        data_dir: &std::path::Path,
        vm_memory_overhead_mib: u64,
    ) -> anyhow::Result<Self> {
        let path = data_dir.join("sessions.json");
        let mut sessions: SessionJournal = read_json_or_default(&path)?;
        for vm_sessions in sessions.values_mut() {
            vm_sessions.retain(|_, session| session.session_idx != BUSY_SESSION_IDX);
            for session in vm_sessions.values_mut() {
                session.state = SessionState::Idle;
            }
        }
        Ok(Self {
            sessions: Arc::new(RwLock::new(sessions)),
            sessions_path: Some(Arc::new(path)),
            vm_memory_overhead_mib,
            ..Self::default()
        })
    }

    /// What a live VM of shape `resources` holds on the host.
    pub fn live_footprint(&self, resources: &Resources) -> Resources {
        Resources {
            memory_mib: resources.memory_mib + self.vm_memory_overhead_mib,
            ..resources.clone()
        }
    }

    /// Registers `requested` as pending demand for as long as the returned
    /// guard is held, and wakes the pressure loop to run an immediate
    /// reclaim pass. Only `admission::acquire` calls this -- it's the one
    /// place a request parks waiting for pool headroom.
    pub fn register_admission_waiter(&self, requested: &Resources) -> AdmissionWaiterGuard {
        self.pending_demand
            .vcpu
            .fetch_add(requested.vcpu, Ordering::SeqCst);
        self.pending_demand
            .memory_mib
            .fetch_add(requested.memory_mib, Ordering::SeqCst);
        self.pending_demand
            .disk_mib
            .fetch_add(requested.disk_mib, Ordering::SeqCst);
        self.admission_requested.notify_one();
        AdmissionWaiterGuard {
            demand: self.pending_demand.clone(),
            requested: requested.clone(),
        }
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
    pub fn capacity_freed(&self) -> tokio::sync::futures::OwnedNotified {
        self.capacity_freed.clone().notified_owned()
    }

    /// Called by every transition that releases pool resources (pause,
    /// suspend, evict) and by VM deletion.
    pub fn notify_capacity_freed(&self) {
        self.capacity_freed.notify_waiters();
    }

    pub async fn insert(&self, record: VmRecord) {
        let mut record = record;
        self.restore_sessions(&mut record).await;
        self.inner
            .write()
            .await
            .insert(record.vm_id.clone(), record);
    }

    async fn restore_sessions(&self, record: &mut VmRecord) {
        if let Some(sessions) = self.sessions.read().await.get(&record.vm_id) {
            record.sessions.clone_from(sessions);
        }
    }

    /// The VM's session at `idx`, made with `cwd` and `env` if it has none
    /// there; without `idx`, a new session at the next index. `None` when
    /// the VM is gone.
    pub async fn open_session(
        &self,
        vm_id: &str,
        idx: Option<u32>,
        cwd: Option<String>,
        env: HashMap<String, String>,
    ) -> std::io::Result<Option<SessionRecord>> {
        let mut map = self.inner.write().await;
        let Some(vm) = map.get_mut(vm_id) else {
            return Ok(None);
        };
        if let Some(existing) = vm.client_sessions().find(|s| Some(s.session_idx) == idx) {
            return Ok(Some(existing.clone()));
        }
        let next_idx = vm
            .client_sessions()
            .map(|s| s.session_idx + 1)
            .max()
            .unwrap_or(0);
        let session = SessionRecord {
            session_id: new_session_id(),
            session_idx: idx.unwrap_or(next_idx),
            cwd,
            env,
            state: SessionState::Idle,
        };
        let mut journal = self.sessions.write().await;
        let mut next = journal.clone();
        next.entry(vm_id.to_owned())
            .or_default()
            .insert(session.session_id.clone(), session.clone());
        self.persist_sessions(&next)?;
        *journal = next;
        drop(journal);
        vm.sessions
            .insert(session.session_id.clone(), session.clone());
        drop(map);
        Ok(Some(session))
    }

    /// Gives a fork its source's sessions: the same indexes, directories and
    /// environments under new ids.
    pub async fn inherit_sessions(&self, vm_id: &str, source: &VmRecord) -> std::io::Result<()> {
        for session in source.client_sessions() {
            self.open_session(
                vm_id,
                Some(session.session_idx),
                session.cwd.clone(),
                session.env.clone(),
            )
            .await?;
        }
        Ok(())
    }

    pub async fn remove_session(&self, vm_id: &str, session_id: &str) -> std::io::Result<bool> {
        let mut map = self.inner.write().await;
        let Some(vm) = map.get_mut(vm_id) else {
            return Ok(false);
        };
        if !vm.sessions.contains_key(session_id) {
            return Ok(false);
        }
        let mut journal = self.sessions.write().await;
        let mut next = journal.clone();
        if let Some(sessions) = next.get_mut(vm_id) {
            sessions.remove(session_id);
        }
        self.persist_sessions(&next)?;
        *journal = next;
        drop(journal);
        vm.sessions.remove(session_id);
        drop(map);
        Ok(true)
    }

    fn persist_sessions(&self, journal: &SessionJournal) -> std::io::Result<()> {
        self.sessions_path
            .as_deref()
            .map_or(Ok(()), |path| write_json_atomic(path, journal))
    }

    pub async fn get(&self, vm_id: &str) -> Option<VmRecord> {
        self.inner.read().await.get(vm_id).cloned()
    }

    pub async fn update<F>(&self, vm_id: &str, f: F) -> bool
    where
        F: FnOnce(&mut VmRecord),
    {
        let mut map = self.inner.write().await;
        map.get_mut(vm_id).is_some_and(|rec| {
            f(rec);
            true
        })
    }

    pub async fn remove(&self, vm_id: &str) -> Option<VmRecord> {
        self.inner.write().await.remove(vm_id)
    }

    async fn replace_portable_sessions(
        &self,
        vm_id: &str,
        sessions: HashMap<String, SessionRecord>,
    ) -> std::io::Result<()> {
        let mut journal = self.sessions.write().await;
        let mut next = journal.clone();
        if sessions.is_empty() {
            next.remove(vm_id);
        } else {
            next.insert(vm_id.to_owned(), sessions.clone());
        }
        self.persist_sessions(&next)?;
        *journal = next;
        drop(journal);
        if let Some(vm) = self.inner.write().await.get_mut(vm_id) {
            vm.sessions = sessions;
        }
        Ok(())
    }

    pub async fn all(&self) -> Vec<VmRecord> {
        self.inner.read().await.values().cloned().collect()
    }

    /// Suspended checkpoints charge their disk image and memory dump.
    const fn suspended_disk_footprint(resources: &Resources) -> u64 {
        resources.disk_mib + resources.memory_mib
    }

    /// Resources currently held across all VMs.
    fn resources_in_use_locked(
        &self,
        map: &HashMap<String, VmRecord>,
    ) -> crate::backend::Resources {
        let mut vcpu = 0u32;
        let mut memory_mib = 0u64;
        let mut disk_mib = 0u64;
        for vm in map.values() {
            match vm.state {
                VmState::Pending | VmState::Running => {
                    let live = self.live_footprint(&vm.resources);
                    vcpu += live.vcpu;
                    memory_mib += live.memory_mib;
                    disk_mib += live.disk_mib;
                }
                VmState::Paused => {
                    let live = self.live_footprint(&vm.resources);
                    memory_mib += live.memory_mib;
                    disk_mib += live.disk_mib;
                }
                VmState::Suspended if vm.snapshot_dir.is_some() => {
                    disk_mib += Self::suspended_disk_footprint(&vm.resources);
                }
                VmState::Suspended => {}
            }
        }
        crate::backend::Resources {
            vcpu,
            memory_mib,
            disk_mib,
        }
    }

    /// Marks `vm_id` busy so no lifecycle transition (pause, suspend) touches
    /// its backend handle until `clear_busy` is called with the returned id.
    /// Uses a running internal session, which `lifecycle::transitions`
    /// already refuses to transition under.
    pub async fn mark_busy(&self, vm_id: &str) -> String {
        let busy_id = format!("busy_{}", Ulid::new());
        self.update(vm_id, |r| {
            r.sessions.insert(
                busy_id.clone(),
                SessionRecord {
                    session_id: busy_id.clone(),
                    session_idx: BUSY_SESSION_IDX,
                    cwd: None,
                    env: HashMap::new(),
                    state: SessionState::Running,
                },
            );
        })
        .await;
        busy_id
    }

    pub async fn clear_busy(&self, vm_id: &str, busy_id: &str) {
        self.update(vm_id, |r| {
            r.sessions.remove(busy_id);
            r.last_active = Instant::now();
        })
        .await;
    }

    pub async fn resources_in_use(&self) -> crate::backend::Resources {
        self.resources_in_use_locked(&*self.inner.read().await)
    }

    /// Reserves a pending VM if its full resource request fits.
    pub async fn try_reserve(
        &self,
        record: &VmRecord,
        total: &Resources,
    ) -> Result<(), crate::backend::Resources> {
        let requested = &self.live_footprint(&record.resources);
        let mut map = self.inner.write().await;
        let in_use = self.resources_in_use_locked(&map);
        let fits = in_use.vcpu + requested.vcpu <= total.vcpu
            && in_use.memory_mib + requested.memory_mib <= total.memory_mib
            && in_use.disk_mib + requested.disk_mib <= total.disk_mib;
        if !fits {
            return Err(in_use);
        }
        map.insert(
            record.vm_id.clone(),
            VmRecord {
                state: VmState::Pending,
                ..record.clone()
            },
        );
        drop(map);
        Ok(())
    }

    /// Reserves the resources needed to resume an existing VM.
    pub async fn try_reserve_resume(
        &self,
        vm_id: &str,
        total: &Resources,
    ) -> Result<(), crate::backend::Resources> {
        let mut map = self.inner.write().await;
        let Some(rec) = map.get(vm_id) else {
            return Ok(());
        };
        let full = self.live_footprint(&rec.resources);
        let held_now = match rec.state {
            VmState::Paused => Resources {
                vcpu: 0,
                memory_mib: full.memory_mib,
                disk_mib: full.disk_mib,
            },
            VmState::Suspended => Resources {
                vcpu: 0,
                memory_mib: 0,
                disk_mib: if rec.snapshot_dir.is_some() {
                    Self::suspended_disk_footprint(&rec.resources)
                } else {
                    0
                },
            },
            VmState::Pending | VmState::Running => full.clone(),
        };
        let in_use = self.resources_in_use_locked(&map);
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

mod runs;
pub use runs::*;
use runs::{read_json_or_default, write_json_atomic, DurableRunRecord};

#[cfg(test)]
mod tests;
