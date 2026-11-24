use std::{
    collections::{HashMap, VecDeque},
    io::Write,
    sync::{Arc, Mutex},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tracing::warn;

use crate::store::{RunRecord, RunState, VmRecord};

use super::client::{CallError, ControlPlane};

mod model;
pub use model::{Activity, WorkerRunRecord, WorkerVmRecord};

const FIRST_RETRY: Duration = Duration::from_secs(1);
const MAX_RETRY: Duration = Duration::from_secs(30);
const COMPACT_AFTER_ACKS: usize = 256;
const MAX_PENDING_AUTHORITATIVE_RECORDS: usize = 256;
const MAX_PENDING_ACTIVITY_RECORDS: usize = 1_024;
const MAX_RECORD_JOURNAL_BYTES: u64 = 128 * 1024 * 1024;
const MAX_DEAD_LETTER_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Delivery {
    PutVm {
        vm_id: String,
        body: WorkerVmRecord,
    },
    DeleteVm {
        vm_id: String,
    },
    PutRun {
        vm_id: String,
        run_id: String,
        body: WorkerRunRecord,
    },
    Activity(Activity),
}

impl Delivery {
    fn same_slot(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::PutVm { vm_id: left, .. } | Self::DeleteVm { vm_id: left },
                Self::PutVm { vm_id: right, .. } | Self::DeleteVm { vm_id: right },
            ) => left == right,
            (
                Self::PutRun {
                    vm_id: left_vm,
                    run_id: left_run,
                    ..
                },
                Self::PutRun {
                    vm_id: right_vm,
                    run_id: right_run,
                    ..
                },
            ) => left_vm == right_vm && left_run == right_run,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
enum LegacyKey {
    Vm(String),
    Run { vm_id: String, run_id: String },
}

#[derive(Deserialize)]
enum LegacyOp {
    PutVm(WorkerVmRecord),
    DeleteVm,
    PutRun(WorkerRunRecord),
}

#[derive(Deserialize)]
struct LegacyPending {
    order: VecDeque<LegacyKey>,
    #[serde(deserialize_with = "deserialize_legacy_ops")]
    ops: HashMap<LegacyKey, LegacyOp>,
}

fn deserialize_legacy_ops<'de, D>(deserializer: D) -> Result<HashMap<LegacyKey, LegacyOp>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Vec::<(LegacyKey, LegacyOp)>::deserialize(deserializer)?
        .into_iter()
        .collect())
}

impl LegacyPending {
    fn into_pending(mut self) -> anyhow::Result<Pending> {
        let mut deliveries = VecDeque::new();
        for key in self.order {
            let Some(operation) = self.ops.remove(&key) else {
                continue;
            };
            deliveries.push_back(match (key, operation) {
                (LegacyKey::Vm(vm_id), LegacyOp::PutVm(body)) => Delivery::PutVm { vm_id, body },
                (LegacyKey::Vm(vm_id), LegacyOp::DeleteVm) => Delivery::DeleteVm { vm_id },
                (LegacyKey::Run { vm_id, run_id }, LegacyOp::PutRun(body)) => Delivery::PutRun {
                    vm_id,
                    run_id,
                    body,
                },
                _ => anyhow::bail!("legacy record key and operation do not match"),
            });
        }
        if !self.ops.is_empty() {
            anyhow::bail!("legacy record file contains unordered operations");
        }
        Ok(Pending {
            deliveries,
            acknowledged: 0,
        })
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Pending {
    deliveries: VecDeque<Delivery>,
    #[serde(skip)]
    acknowledged: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum JournalEntry {
    Enqueue(Box<Delivery>),
    Ack,
}

struct Queue {
    pending: Mutex<Pending>,
    journal: std::path::PathBuf,
    dead_letters: std::path::PathBuf,
    changed: Notify,
}

#[derive(Serialize)]
struct DeadLetter<'a> {
    delivery: &'a Delivery,
    status: u16,
    body: &'a str,
}

#[derive(Clone)]
pub struct Records {
    queue: Arc<Queue>,
    control_plane: ControlPlane,
}

impl Records {
    pub fn new(data_dir: &std::path::Path, control_plane: ControlPlane) -> anyhow::Result<Self> {
        let journal = data_dir.join("pending_records.jsonl");
        let dead_letters = data_dir.join("rejected_records.jsonl");
        let legacy = data_dir.join("pending_records.json");
        let mut pending = load_journal(&journal, &legacy)?;
        if pending
            .deliveries
            .iter()
            .filter(|delivery| !matches!(delivery, Delivery::Activity(_)))
            .count()
            > MAX_PENDING_AUTHORITATIVE_RECORDS
            || pending
                .deliveries
                .iter()
                .filter(|delivery| matches!(delivery, Delivery::Activity(_)))
                .count()
                > MAX_PENDING_ACTIVITY_RECORDS
        {
            anyhow::bail!("control-plane record queue exceeds its durable limit");
        }
        if control_plane.incarnation().is_some() {
            pending.discard_ownership();
        }
        compact(&journal, &pending)?;
        if legacy.exists() {
            std::fs::remove_file(&legacy)?;
        }
        Ok(Self {
            queue: Arc::new(Queue {
                pending: Mutex::new(pending),
                journal,
                dead_letters,
                changed: Notify::new(),
            }),
            control_plane,
        })
    }

    fn push(&self, delivery: Delivery) -> std::io::Result<()> {
        let mut pending = self
            .queue
            .pending
            .lock()
            .map_err(|_| std::io::Error::other("records lock poisoned"))?;
        if pending.deliveries.contains(&delivery) {
            return Ok(());
        }
        let activity = matches!(&delivery, Delivery::Activity(_));
        let mut next = pending.deliveries.clone();
        enqueue(&mut next, delivery.clone());
        let limit = if activity {
            MAX_PENDING_ACTIVITY_RECORDS
        } else {
            MAX_PENDING_AUTHORITATIVE_RECORDS
        };
        let queued = next
            .iter()
            .filter(|queued| matches!(queued, Delivery::Activity(_)) == activity)
            .count();
        if queued > limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "control-plane record queue is full",
            ));
        }
        let entry = JournalEntry::Enqueue(Box::new(delivery));
        let entry_bytes = serde_json::to_vec(&entry)
            .map_err(std::io::Error::other)?
            .len()
            .saturating_add(1) as u64;
        let journal_bytes = self
            .queue
            .journal
            .metadata()
            .map_or(0, |metadata| metadata.len());
        let compacted = journal_bytes.saturating_add(entry_bytes) > MAX_RECORD_JOURNAL_BYTES;
        if compacted {
            let compacted = Pending {
                deliveries: next.clone(),
                acknowledged: 0,
            };
            compact(&self.queue.journal, &compacted)?;
        } else {
            append(&self.queue.journal, &entry)?;
        }
        pending.deliveries = next;
        if compacted {
            pending.acknowledged = 0;
        }
        drop(pending);
        self.queue.changed.notify_waiters();
        Ok(())
    }

    pub fn vm(&self, vm: &VmRecord) -> std::io::Result<()> {
        self.push(Delivery::PutVm {
            vm_id: vm.vm_id.clone(),
            body: WorkerVmRecord::new(vm),
        })
    }

    pub async fn vm_deleted(&self, vm_id: &str) -> Result<(), CallError> {
        self.control_plane.ready().await;
        self.control_plane.delete(&format!("vms/{vm_id}")).await
    }

    pub fn run(&self, run_id: &str, run: &RunRecord) -> std::io::Result<()> {
        self.run_record(run_id, run)?;
        if let Err(error) = self.activity(Activity::run(run_id, run, RunState::Running)) {
            warn!(run_id, err = %error, "persist run activity failed");
        }
        if run.state != RunState::Running {
            if let Err(error) = self.activity(Activity::run(run_id, run, run.state)) {
                warn!(run_id, err = %error, "persist completed run activity failed");
            }
        }
        Ok(())
    }

    pub fn run_record(&self, run_id: &str, run: &RunRecord) -> std::io::Result<()> {
        self.push(Delivery::PutRun {
            vm_id: run.vm_id.clone(),
            run_id: run_id.to_owned(),
            body: WorkerRunRecord::new(run),
        })
    }

    pub fn activity(&self, activity: Activity) -> std::io::Result<()> {
        self.push(Delivery::Activity(activity))
    }

    pub fn abandon_vm(&self, vm_id: &str) -> std::io::Result<()> {
        let mut pending = self
            .queue
            .pending
            .lock()
            .map_err(|_| std::io::Error::other("records lock poisoned"))?;
        pending.deliveries.retain(
            |delivery| !matches!(delivery, Delivery::PutVm { vm_id: queued, .. } if queued == vm_id),
        );
        compact(&self.queue.journal, &pending)?;
        pending.acknowledged = 0;
        drop(pending);
        self.queue.changed.notify_waiters();
        Ok(())
    }

    pub async fn flush(&self, timeout: Duration) -> bool {
        self.flush_matching(timeout, |_| true).await
    }

    /// Waits for the VM's queued VM and run records to be delivered.
    pub async fn flush_vm(&self, vm_id: &str, timeout: Duration) -> bool {
        self.flush_matching(timeout, |delivery| match delivery {
            Delivery::PutVm { vm_id: queued, .. }
            | Delivery::DeleteVm { vm_id: queued }
            | Delivery::PutRun { vm_id: queued, .. } => queued == vm_id,
            Delivery::Activity(_) => false,
        })
        .await
    }

    async fn flush_matching(&self, timeout: Duration, matches: impl Fn(&Delivery) -> bool) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let changed = self.queue.changed.notified();
            let empty = match self.queue.pending.lock() {
                Ok(pending) => !pending.deliveries.iter().any(&matches),
                Err(_) => return false,
            };
            if empty {
                return true;
            }
            if tokio::time::timeout_at(deadline, changed).await.is_err() {
                return false;
            }
        }
    }

    pub async fn send_loop(self) {
        self.control_plane.ready().await;
        let mut retry = FIRST_RETRY;
        loop {
            let changed = self.queue.changed.notified();
            let delivery = if let Ok(pending) = self.queue.pending.lock() {
                pending.deliveries.front().cloned()
            } else {
                warn!("records lock poisoned; stopping delivery loop");
                return;
            };
            let Some(delivery) = delivery else {
                changed.await;
                continue;
            };
            match send(&self.control_plane, &delivery).await {
                Ok(()) => {
                    if let Err(error) = self.acknowledge(&delivery) {
                        warn!(?delivery, err = %error, "persist control-plane acknowledgement failed");
                        tokio::time::sleep(retry).await;
                        retry = (retry * 2).min(MAX_RETRY);
                    } else {
                        retry = FIRST_RETRY;
                    }
                }
                Err(CallError::Rejected { status, body }) => {
                    if self
                        .queue
                        .dead_letters
                        .metadata()
                        .is_ok_and(|metadata| metadata.len() >= MAX_DEAD_LETTER_BYTES)
                    {
                        if let Err(error) = std::fs::remove_file(&self.queue.dead_letters) {
                            warn!(err = %error, "rotate rejected control-plane records failed");
                        }
                    }
                    if let Err(error) = append_json_line(
                        &self.queue.dead_letters,
                        &DeadLetter {
                            delivery: &delivery,
                            status: status.as_u16(),
                            body: &body,
                        },
                    ) {
                        warn!(?delivery, err = %error, "persist rejected control-plane record failed");
                        tokio::time::sleep(retry).await;
                        retry = (retry * 2).min(MAX_RETRY);
                        continue;
                    }
                    warn!(?delivery, %status, body, "control-plane delivery rejected");
                    if let Err(error) = self.acknowledge(&delivery) {
                        warn!(?delivery, err = %error, "persist control-plane acknowledgement failed");
                        tokio::time::sleep(retry).await;
                        retry = (retry * 2).min(MAX_RETRY);
                    } else {
                        retry = FIRST_RETRY;
                    }
                }
                Err(error @ CallError::Transient(_)) => {
                    warn!(?delivery, err = %error, retry_in = ?retry, "control-plane delivery failed");
                    tokio::time::sleep(retry).await;
                    retry = (retry * 2).min(MAX_RETRY);
                }
            }
        }
    }

    fn acknowledge(&self, delivery: &Delivery) -> std::io::Result<()> {
        let mut pending = self
            .queue
            .pending
            .lock()
            .map_err(|_| std::io::Error::other("records lock poisoned"))?;
        if pending.deliveries.front() != Some(delivery) {
            return Ok(());
        }
        append(&self.queue.journal, &JournalEntry::Ack)?;
        pending.deliveries.pop_front();
        pending.acknowledged += 1;
        if pending.acknowledged >= COMPACT_AFTER_ACKS {
            match compact(&self.queue.journal, &pending) {
                Ok(()) => pending.acknowledged = 0,
                Err(error) => warn!(err = %error, "compact pending control-plane records failed"),
            }
        }
        drop(pending);
        self.queue.changed.notify_waiters();
        Ok(())
    }
}

impl Pending {
    fn discard_ownership(&mut self) {
        self.deliveries.retain(|delivery| {
            !matches!(delivery, Delivery::PutVm { .. } | Delivery::DeleteVm { .. })
        });
    }
}

fn load_journal(path: &std::path::Path, legacy: &std::path::Path) -> anyhow::Result<Pending> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return match std::fs::read(legacy) {
                Ok(bytes) => {
                    let pending: LegacyPending = serde_json::from_slice(&bytes)
                        .map_err(|error| anyhow::anyhow!("read {}: {error}", legacy.display()))?;
                    pending
                        .into_pending()
                        .map_err(|error| anyhow::anyhow!("read {}: {error}", legacy.display()))
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    Ok(Pending::default())
                }
                Err(error) => Err(anyhow::anyhow!("read {}: {error}", legacy.display())),
            };
        }
        Err(error) => return Err(anyhow::anyhow!("read {}: {error}", path.display())),
    };
    let mut pending = Pending::default();
    let complete_end = if bytes.last() == Some(&b'\n') {
        bytes.len().saturating_sub(1)
    } else {
        bytes.iter().rposition(|byte| *byte == b'\n').unwrap_or(0)
    };
    let complete = &bytes[..complete_end];
    for (index, line) in complete.split(|byte| *byte == b'\n').enumerate() {
        if line.is_empty() {
            continue;
        }
        let entry: JournalEntry = serde_json::from_slice(line).map_err(|error| {
            anyhow::anyhow!("read {} line {}: {error}", path.display(), index + 1)
        })?;
        match entry {
            JournalEntry::Enqueue(delivery) => enqueue(&mut pending.deliveries, *delivery),
            JournalEntry::Ack => {
                pending.deliveries.pop_front().ok_or_else(|| {
                    anyhow::anyhow!(
                        "read {} line {}: acknowledgement without record",
                        path.display(),
                        index + 1
                    )
                })?;
            }
        }
    }
    Ok(pending)
}

fn enqueue(deliveries: &mut VecDeque<Delivery>, delivery: Delivery) {
    if let Some(existing) = deliveries
        .iter_mut()
        .find(|existing| existing.same_slot(&delivery))
    {
        *existing = delivery;
    } else {
        deliveries.push_back(delivery);
    }
}

fn append(path: &std::path::Path, entry: &JournalEntry) -> std::io::Result<()> {
    append_json_line(path, entry)
}

fn append_json_line(path: &std::path::Path, value: &impl Serialize) -> std::io::Result<()> {
    let created = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    serde_json::to_writer(&mut file, value).map_err(std::io::Error::other)?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    if created {
        let parent = path
            .parent()
            .ok_or_else(|| std::io::Error::other("record path has no parent"))?;
        sync_journal_parent(parent)?;
    }
    Ok(())
}

fn compact(path: &std::path::Path, pending: &Pending) -> std::io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("records journal has no parent"))?;
    let tmp = parent.join(format!(".pending-records-{}.tmp", ulid::Ulid::new()));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&tmp)?;
    for delivery in &pending.deliveries {
        serde_json::to_writer(
            &mut file,
            &JournalEntry::Enqueue(Box::new(delivery.clone())),
        )
        .map_err(std::io::Error::other)?;
        file.write_all(b"\n")?;
    }
    file.sync_all()?;
    replace_journal(&tmp, path)?;
    sync_journal_parent(parent)
}

#[cfg(not(windows))]
fn replace_journal(source: &std::path::Path, destination: &std::path::Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_journal(source: &std::path::Path, destination: &std::path::Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };
    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let replaced = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn sync_journal_parent(parent: &std::path::Path) -> std::io::Result<()> {
    std::fs::File::open(parent)?.sync_all()
}

#[cfg(windows)]
fn sync_journal_parent(_: &std::path::Path) -> std::io::Result<()> {
    Ok(())
}

async fn send(control_plane: &ControlPlane, delivery: &Delivery) -> Result<(), CallError> {
    match delivery {
        Delivery::PutVm { vm_id, body } => control_plane.put(&format!("vms/{vm_id}"), body).await,
        Delivery::DeleteVm { vm_id } => control_plane.delete(&format!("vms/{vm_id}")).await,
        Delivery::PutRun {
            vm_id,
            run_id,
            body,
        } => {
            control_plane
                .put(&format!("vms/{vm_id}/runs/{run_id}"), body)
                .await
        }
        Delivery::Activity(activity) => control_plane.activity(activity).await,
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests;
