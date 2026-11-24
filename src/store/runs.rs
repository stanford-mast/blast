use super::*;

pub fn new_vm_id() -> String {
    format!("vm_{}", Ulid::new())
}
pub fn new_session_id() -> String {
    format!("sess_{}", Ulid::new())
}
pub fn new_run_id() -> String {
    format!("run_{}", Ulid::new())
}

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
    pub vm_id: String,
    pub session_id: Option<String>,
    pub command: String,
    pub started_at: SystemTime,
    pub completed_at: Option<SystemTime>,
    pub state: RunState,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub stdout_encoding: Option<&'static str>,
    pub stderr_encoding: Option<&'static str>,
    pub exit_code: Option<i32>,
    pub fail_reason: Option<String>,
    /// Whether the command printed more than the output kept.
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    /// The request's idempotency key and a digest of what it asked for.
    pub idempotency: Option<(String, String)>,
}

impl RunRecord {
    pub fn started(vm_id: String, session_id: Option<String>, command: String) -> Self {
        Self {
            stdout_truncated: false,
            stderr_truncated: false,
            idempotency: None,
            vm_id,
            session_id,
            command,
            started_at: SystemTime::now(),
            completed_at: None,
            state: RunState::Running,
            stdout: None,
            stderr: None,
            stdout_encoding: None,
            stderr_encoding: None,
            exit_code: None,
            fail_reason: None,
        }
    }
}

/// Poll store for run results, lets a caller that got back a `"running"`
/// state (because `time_to_background` elapsed before the command finished)
/// retrieve the eventual result via `GET /v1/vms/:vm_id/runs/:run_id`.
#[derive(Clone, Default)]
pub struct RunStore {
    inner: Arc<RwLock<HashMap<String, RunRecord>>>,
    path: Option<Arc<PathBuf>>,
    /// Notified whenever a run changes or is forgotten.
    changed: Arc<Notify>,
}

pub(super) const MAX_LOCAL_RUNS_PER_VM: usize = 4_096;
pub(super) const MAX_LOCAL_RUNS_TOTAL: usize = 16_384;

impl RunStore {
    pub fn durable(data_dir: &std::path::Path) -> anyhow::Result<Self> {
        let path = data_dir.join("runs.json");
        let mut records: HashMap<String, DurableRunRecord> = read_json_or_default(&path)?;
        let now = SystemTime::now();
        let mut records: HashMap<_, _> = records
            .drain()
            .map(|(id, record)| {
                let mut record = RunRecord::from(record);
                if record.state == RunState::Running {
                    record.state = RunState::Failed;
                    record.completed_at = Some(now);
                    record.fail_reason =
                        Some("worker restarted while the command was running".into());
                }
                (id, record)
            })
            .collect();
        let vm_ids: std::collections::HashSet<_> =
            records.values().map(|run| run.vm_id.clone()).collect();
        for vm_id in vm_ids {
            prune_terminal_runs(&mut records, &vm_id)?;
        }
        let store = Self {
            inner: Arc::new(RwLock::new(records)),
            path: Some(Arc::new(path)),
            changed: Arc::default(),
        };
        let records = store
            .inner
            .try_read()
            .map_err(|_| std::io::Error::other("new run store is locked"))?;
        store.persist(&records)?;
        drop(records);
        Ok(store)
    }

    pub async fn insert(&self, run_id: String, record: RunRecord) -> std::io::Result<()> {
        let mut records = self.inner.write().await;
        let mut next = records.clone();
        let vm_id = record.vm_id.clone();
        next.insert(run_id, record);
        prune_terminal_runs(&mut next, &vm_id)?;
        self.persist(&next)?;
        *records = next;
        drop(records);
        Ok(())
    }

    pub async fn get(&self, run_id: &str) -> Option<RunRecord> {
        self.inner.read().await.get(run_id).cloned()
    }

    pub async fn update<F>(&self, run_id: &str, f: F) -> std::io::Result<bool>
    where
        F: FnOnce(&mut RunRecord),
    {
        let mut records = self.inner.write().await;
        let mut next = records.clone();
        let updated = next.get_mut(run_id).is_some_and(|rec| {
            f(rec);
            true
        });
        if updated {
            let vm_id = next
                .get(run_id)
                .map(|run| run.vm_id.clone())
                .ok_or_else(|| std::io::Error::other("updated run vanished"))?;
            prune_terminal_runs(&mut next, &vm_id)?;
            self.persist(&next)?;
            *records = next;
        }
        drop(records);
        self.changed.notify_waiters();
        Ok(updated)
    }

    /// The run once it is no longer running; `None` if it is forgotten first.
    pub async fn ended(&self, run_id: &str) -> Option<RunRecord> {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            // Registered before the read, so no change between the two is missed.
            changed.as_mut().enable();
            match self.get(run_id).await {
                Some(run) if run.state == RunState::Running => changed.await,
                run => return run,
            }
        }
    }

    /// The VM's run that `key` started, if any.
    pub async fn keyed(&self, vm_id: &str, key: &str) -> Option<(String, RunRecord)> {
        self.inner
            .read()
            .await
            .iter()
            .find(|(_, run)| {
                run.vm_id == vm_id && run.idempotency.as_ref().is_some_and(|(k, _)| k == key)
            })
            .map(|(id, run)| (id.clone(), run.clone()))
    }

    /// The VM's runs, oldest first.
    pub async fn of_vm(&self, vm_id: &str) -> Vec<(String, RunRecord)> {
        let mut runs: Vec<_> = self
            .inner
            .read()
            .await
            .iter()
            .filter(|(_, run)| run.vm_id == vm_id)
            .map(|(id, run)| (id.clone(), run.clone()))
            .collect();
        runs.sort_by(|a, b| (a.1.started_at, &a.0).cmp(&(b.1.started_at, &b.0)));
        runs
    }

    pub async fn running_count(&self) -> usize {
        self.inner
            .read()
            .await
            .values()
            .filter(|r| r.state == RunState::Running)
            .count()
    }

    pub async fn all(&self) -> Vec<(String, RunRecord)> {
        self.inner
            .read()
            .await
            .iter()
            .map(|(id, run)| (id.clone(), run.clone()))
            .collect()
    }

    async fn replace_vm_runs(
        &self,
        vm_id: &str,
        incoming: HashMap<String, DurableRunRecord>,
    ) -> std::io::Result<()> {
        let mut records = self.inner.write().await;
        let mut next = records.clone();
        next.retain(|_, run| run.vm_id != vm_id);
        next.extend(incoming.into_iter().map(|(id, run)| (id, run.into())));
        prune_terminal_runs(&mut next, vm_id)?;
        self.persist(&next)?;
        *records = next;
        drop(records);
        self.changed.notify_waiters();
        Ok(())
    }

    fn persist(&self, records: &HashMap<String, RunRecord>) -> std::io::Result<()> {
        let Some(path) = self.path.as_deref() else {
            return Ok(());
        };
        let durable: HashMap<_, _> = records
            .iter()
            .map(|(id, run)| (id, DurableRunRecord::from(run)))
            .collect();
        write_json_atomic(path, &durable)
    }
}

pub(super) fn prune_terminal_runs(
    records: &mut HashMap<String, RunRecord>,
    vm_id: &str,
) -> std::io::Result<()> {
    let vm_runs = records.values().filter(|run| run.vm_id == vm_id).count();
    let remove = vm_runs.saturating_sub(MAX_LOCAL_RUNS_PER_VM);
    let mut terminal: Vec<_> = records
        .iter()
        .filter(|(_, run)| run.vm_id == vm_id && run.state != RunState::Running)
        .map(|(id, run)| (id.clone(), run.completed_at.unwrap_or(run.started_at)))
        .collect();
    terminal.sort_by_key(|(_, completed_at)| *completed_at);
    if terminal.len() < remove {
        return Err(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "too many active runs to retain another local run",
        ));
    }
    for (id, _) in terminal.into_iter().take(remove) {
        records.remove(&id);
    }
    let remove = records.len().saturating_sub(MAX_LOCAL_RUNS_TOTAL);
    let mut terminal: Vec<_> = records
        .iter()
        .filter(|(_, run)| run.state != RunState::Running)
        .map(|(id, run)| (id.clone(), run.completed_at.unwrap_or(run.started_at)))
        .collect();
    terminal.sort_by_key(|(_, completed_at)| *completed_at);
    if terminal.len() < remove {
        return Err(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "too many active runs to retain another local run",
        ));
    }
    // The router answers BLAST run reads from its durable run record. This
    // worker copy is only the bounded cache used by standalone mode and by a
    // synchronous response while its authoritative delivery is queued.
    for (id, _) in terminal.into_iter().take(remove) {
        records.remove(&id);
    }
    Ok(())
}

pub async fn write_portable_vm_state(
    path: &std::path::Path,
    vm_id: &str,
    store: &Store,
    runs: &RunStore,
) -> anyhow::Result<()> {
    let sessions = store
        .get(vm_id)
        .await
        .map(|vm| {
            vm.sessions
                .into_iter()
                .filter(|(_, session)| session.session_idx != BUSY_SESSION_IDX)
                .collect()
        })
        .unwrap_or_default();
    let runs = runs
        .all()
        .await
        .into_iter()
        .filter(|(_, run)| run.vm_id == vm_id)
        .map(|(id, run)| (id, DurableRunRecord::from(&run)))
        .collect();
    crate::durable::write_json_atomic(path, &PortableVmState { sessions, runs })?;
    Ok(())
}

pub async fn restore_portable_vm_state(
    path: &std::path::Path,
    vm_id: &str,
    store: &Store,
    runs: &RunStore,
) -> anyhow::Result<()> {
    let bytes = std::fs::read(path)?;
    let state: PortableVmState = serde_json::from_slice(&bytes)?;
    if state
        .runs
        .values()
        .any(|run| run.vm_id != vm_id || run.state == RunState::Running)
    {
        anyhow::bail!("portable VM state contains an invalid run");
    }
    store
        .replace_portable_sessions(vm_id, state.sessions)
        .await?;
    runs.replace_vm_runs(vm_id, state.runs).await?;
    Ok(())
}

pub async fn forget_portable_vm_state(
    vm_id: &str,
    store: &Store,
    runs: &RunStore,
) -> anyhow::Result<()> {
    store
        .replace_portable_sessions(vm_id, HashMap::new())
        .await?;
    runs.replace_vm_runs(vm_id, HashMap::new()).await?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
pub(super) struct DurableRunRecord {
    vm_id: String,
    session_id: Option<String>,
    command: String,
    started_at: SystemTime,
    completed_at: Option<SystemTime>,
    state: RunState,
    stdout: Option<String>,
    stderr: Option<String>,
    stdout_encoding: Option<String>,
    stderr_encoding: Option<String>,
    exit_code: Option<i32>,
    fail_reason: Option<String>,
    #[serde(default)]
    stdout_truncated: bool,
    #[serde(default)]
    stderr_truncated: bool,
    #[serde(default)]
    idempotency: Option<(String, String)>,
}

impl From<&RunRecord> for DurableRunRecord {
    fn from(r: &RunRecord) -> Self {
        Self {
            vm_id: r.vm_id.clone(),
            session_id: r.session_id.clone(),
            command: r.command.clone(),
            started_at: r.started_at,
            completed_at: r.completed_at,
            state: r.state,
            stdout: r.stdout.clone(),
            stderr: r.stderr.clone(),
            stdout_encoding: r.stdout_encoding.map(str::to_owned),
            stderr_encoding: r.stderr_encoding.map(str::to_owned),
            exit_code: r.exit_code,
            fail_reason: r.fail_reason.clone(),
            stdout_truncated: r.stdout_truncated,
            stderr_truncated: r.stderr_truncated,
            idempotency: r.idempotency.clone(),
        }
    }
}
impl From<DurableRunRecord> for RunRecord {
    fn from(r: DurableRunRecord) -> Self {
        Self {
            vm_id: r.vm_id,
            session_id: r.session_id,
            command: r.command,
            started_at: r.started_at,
            completed_at: r.completed_at,
            state: r.state,
            stdout: r.stdout,
            stderr: r.stderr,
            stdout_encoding: durable_encoding(r.stdout_encoding.as_deref()),
            stderr_encoding: durable_encoding(r.stderr_encoding.as_deref()),
            exit_code: r.exit_code,
            fail_reason: r.fail_reason,
            stdout_truncated: r.stdout_truncated,
            stderr_truncated: r.stderr_truncated,
            idempotency: r.idempotency,
        }
    }
}

fn durable_encoding(value: Option<&str>) -> Option<&'static str> {
    match value {
        Some("utf-8") => Some("utf-8"),
        Some("base64") => Some("base64"),
        _ => None,
    }
}

pub(super) fn read_json_or_default<T: serde::de::DeserializeOwned + Default>(
    path: &std::path::Path,
) -> anyhow::Result<T> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(Into::into),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(e) => Err(e.into()),
    }
}
pub(super) fn write_json_atomic<T: Serialize>(
    path: &std::path::Path,
    value: &T,
) -> std::io::Result<()> {
    crate::durable::write_json_atomic(path, value)
}
