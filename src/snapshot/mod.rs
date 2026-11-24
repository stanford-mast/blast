//! Local and durable VM snapshots.

pub mod s3;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak},
    time::{Duration, Instant, SystemTime},
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::{fs, io::AsyncWriteExt, sync::RwLock};

use crate::backend::Resources;

/// What a worker needs to take over a suspended VM it never saw.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VmManifest {
    pub vm_id: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub image: Option<String>,
    pub parent_vm_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workdir: Option<String>,
    /// Seconds since the Unix epoch.
    pub created_at: u64,
    pub resources: Resources,
    /// Host-local backend handle. It lets a restarted worker reattach to a
    /// VM owned by a daemon-backed hypervisor. Another worker safely ignores
    /// it because the handle will not exist there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_handle: Option<String>,
}

impl VmManifest {
    pub fn created_at(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(self.created_at)
    }
}

/// Where durable snapshot copies go.
pub enum Storage {
    /// No durable copy: the local snapshot is the only one.
    None,
    /// A directory, typically a mount of shared storage.
    Dir(PathBuf),
    /// Object storage; the bucket and its short-lived credentials arrive
    /// with each heartbeat, so there is none until the first answer.
    Object {
        bucket: RwLock<Option<Box<s3::Bucket>>>,
        endpoint: Option<String>,
    },
}

impl Storage {
    pub const fn object(endpoint: Option<String>) -> Self {
        Self::Object {
            bucket: RwLock::const_new(None),
            endpoint,
        }
    }
}

pub struct SnapshotStore {
    root: PathBuf,
    storage: Storage,
    shared_storage: bool,
    /// Each running VM's latest fork checkpoint and when it began, kept
    /// while a fork still reads it.
    fork_checkpoints: Mutex<HashMap<String, (Instant, Weak<TransientSnapshot>)>>,
    /// Each VM's pin of its local snapshot, kept while a fork still reads it.
    /// Held while a pin is made, so two forks never link the same files at
    /// once.
    local_pins: tokio::sync::Mutex<HashMap<String, Weak<TransientSnapshot>>>,
}

pub struct LastSnapshot {
    /// The local snapshot directory, when this worker has one.
    pub local: Option<PathBuf>,
    /// Whether a stored generation is selected for the VM.
    pub stored: bool,
}

pub struct TransientSnapshot {
    path: PathBuf,
}

impl TransientSnapshot {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TransientSnapshot {
    fn drop(&mut self) {
        match std::fs::remove_dir_all(&self.path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!(
                path = %self.path.display(),
                err = %error,
                "removing transient snapshot failed"
            ),
        }
    }
}

/// Filename for the backend-kind tag `write_backend_tag`/`check_backend_tag`
/// share. Lives inside the snapshot directory so it travels with it.
const BACKEND_TAG_FILE: &str = ".blast-backend";
/// Filename of the [`VmManifest`] inside the snapshot directory.
const MANIFEST_FILE: &str = ".blast-vm.json";
const COMMIT_FILE: &str = ".blast-commit.json";
const COMMITS_DIR: &str = ".blast-commits";
const CLAIM_TOKEN_FILE: &str = ".blast-claim-generation";
/// Names the stored generation a local snapshot directory was restored
/// from; absent once the directory holds anything else.
const LOCAL_GENERATION_FILE: &str = ".blast-generation";
pub const PORTABLE_STATE_FILE: &str = ".blast-state.json";
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingGeneration {
    pub vm_generation: i64,
    pub snapshot_id: String,
    #[serde(default)]
    pub uploaded: bool,
    #[serde(default)]
    pub kind: PendingKind,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingKind {
    #[default]
    Suspended,
    DirtyCheckpoint,
}

#[derive(Debug, Serialize, Deserialize)]
struct SnapshotCommit {
    generation: String,
    previous: Option<String>,
    files: BTreeMap<String, String>,
}

/// VM ids name directories and object keys, so only a conservative alphabet
/// is ever used to build a path from one.
pub fn is_safe_vm_id(vm_id: &str) -> bool {
    !vm_id.is_empty()
        && vm_id.len() <= 128
        && vm_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn check_vm_id(vm_id: &str) -> Result<()> {
    if !is_safe_vm_id(vm_id) {
        bail!("invalid VM id {vm_id:?}");
    }
    Ok(())
}

impl SnapshotStore {
    pub fn new(root: impl Into<PathBuf>, storage: Storage) -> Self {
        let shared_storage = matches!(&storage, Storage::Object { .. });
        Self::with_shared_storage(root, storage, shared_storage)
    }

    pub fn with_shared_storage(
        root: impl Into<PathBuf>,
        storage: Storage,
        shared_storage: bool,
    ) -> Self {
        Self {
            root: root.into(),
            storage,
            shared_storage,
            fork_checkpoints: Mutex::default(),
            local_pins: tokio::sync::Mutex::default(),
        }
    }

    /// A fork checkpoint of `vm_id` begun at or after `since` that is still
    /// being read.
    pub fn fork_checkpoint_since(
        &self,
        vm_id: &str,
        since: Instant,
    ) -> Option<Arc<TransientSnapshot>> {
        self.fork_checkpoints
            .lock()
            .expect("fork checkpoints poisoned")
            .get(vm_id)
            .filter(|(started, _)| *started >= since)
            .and_then(|(_, checkpoint)| checkpoint.upgrade())
    }

    /// Offers a checkpoint of `vm_id` begun at `started` to later forks.
    pub fn share_fork_checkpoint(
        &self,
        vm_id: &str,
        started: Instant,
        checkpoint: &Arc<TransientSnapshot>,
    ) {
        let mut checkpoints = self
            .fork_checkpoints
            .lock()
            .expect("fork checkpoints poisoned");
        checkpoints.retain(|_, (_, checkpoint)| checkpoint.strong_count() > 0);
        checkpoints.insert(vm_id.to_owned(), (started, Arc::downgrade(checkpoint)));
    }

    /// Whether a snapshot can be stored durably right now: object storage
    /// needs a bucket from the control plane first.
    pub async fn can_store(&self) -> bool {
        match &self.storage {
            Storage::None => false,
            Storage::Dir(_) => true,
            Storage::Object { bucket, .. } => bucket.read().await.is_some(),
        }
    }

    /// Takes the bucket and credentials from the latest heartbeat. Ignored
    /// unless this worker stores snapshots in object storage.
    pub async fn set_object_storage(&self, storage: s3::ObjectStorage) {
        let Storage::Object { bucket, endpoint } = &self.storage else {
            return;
        };
        let mut current = bucket.write().await;
        // Unchanged credentials keep the client and its open connections.
        if current.as_deref().map(s3::Bucket::storage) != Some(&storage) {
            *current = Some(Box::new(s3::Bucket::new(storage, endpoint.clone())));
        }
    }

    fn snapshots_dir(&self) -> PathBuf {
        self.root.join("snapshots")
    }

    fn transient_snapshots_dir(&self) -> PathBuf {
        self.root.join("transient-snapshots")
    }

    pub async fn transient_snapshot(&self, vm_id: &str) -> Result<TransientSnapshot> {
        check_vm_id(vm_id)?;
        let path = self
            .transient_snapshots_dir()
            .join(format!("{vm_id}-{}", ulid::Ulid::new()));
        fs::create_dir_all(&path).await?;
        Ok(TransientSnapshot { path })
    }

    pub async fn scavenge_transient_snapshots(&self) -> Result<()> {
        remove_dir_if_exists(&self.transient_snapshots_dir()).await
    }

    /// Local path for a VM's snapshot directory.
    pub fn snap_dir(&self, vm_id: &str) -> PathBuf {
        self.snapshots_dir().join(vm_id)
    }

    fn claim_dir(&self, vm_id: &str, generation: i64) -> PathBuf {
        self.root
            .join("claims")
            .join(vm_id)
            .join(generation.to_string())
    }

    fn removals_dir(&self) -> PathBuf {
        self.root.join("snapshot-removals")
    }

    fn owned_dir(&self) -> PathBuf {
        self.root.join("owned-vms")
    }

    fn selected_snapshots_dir(&self) -> PathBuf {
        self.root.join("selected-snapshots")
    }

    fn pending_generation_path(&self, vm_id: &str) -> PathBuf {
        self.root.join("pending-snapshots").join(vm_id)
    }

    fn local_discards_dir(&self) -> PathBuf {
        self.root.join("local-discards")
    }

    pub async fn begin_local_discard(&self, vm_id: &str) -> Result<()> {
        check_vm_id(vm_id)?;
        let directory = self.local_discards_dir();
        fs::create_dir_all(&directory).await?;
        let tombstone = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(directory.join(vm_id))
            .await?;
        tombstone.sync_all().await?;
        sync_directory(&directory).await?;
        Ok(())
    }

    pub async fn pending_local_discards(&self) -> Result<Vec<String>> {
        let mut pending = Vec::new();
        let mut entries = match fs::read_dir(self.local_discards_dir()).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(pending),
            Err(error) => return Err(error).context("read local VM discards"),
        };
        while let Some(entry) = entries.next_entry().await? {
            if let Some(vm_id) = entry.file_name().to_str().filter(|id| is_safe_vm_id(id)) {
                pending.push(vm_id.to_owned());
            }
        }
        pending.sort();
        Ok(pending)
    }

    pub async fn finish_local_discard(&self, vm_id: &str) -> Result<()> {
        check_vm_id(vm_id)?;
        let directory = self.local_discards_dir();
        match fs::remove_file(directory.join(vm_id)).await {
            Ok(()) => sync_directory(&directory).await?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("finish local VM discard"),
        }
        Ok(())
    }

    pub async fn pending_generation(&self, vm_id: &str) -> Result<Option<PendingGeneration>> {
        check_vm_id(vm_id)?;
        match fs::read(self.pending_generation_path(vm_id)).await {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .context("parse pending snapshot generation")
                .map(Some),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).context("read pending snapshot generation"),
        }
    }

    pub async fn set_pending_generation(
        &self,
        vm_id: &str,
        pending: &PendingGeneration,
    ) -> Result<()> {
        check_vm_id(vm_id)?;
        if ulid::Ulid::from_string(&pending.snapshot_id).is_err() {
            bail!("invalid snapshot generation");
        }
        let path = self.pending_generation_path(vm_id);
        fs::create_dir_all(path.parent().expect("pending snapshot parent")).await?;
        write_atomic_json(&path, pending).await
    }

    pub async fn mark_generation_uploaded(&self, vm_id: &str) -> Result<()> {
        let Some(mut pending) = self.pending_generation(vm_id).await? else {
            bail!("no pending snapshot generation for {vm_id}");
        };
        pending.uploaded = true;
        self.set_pending_generation(vm_id, &pending).await
    }

    pub async fn finish_generation(&self, vm_id: &str, generation: &str) -> Result<()> {
        self.select_generation(vm_id, generation).await?;
        remove_file_if_exists(&self.pending_generation_path(vm_id)).await
    }

    pub async fn abandon_generation(&self, vm_id: &str) -> Result<()> {
        check_vm_id(vm_id)?;
        remove_file_if_exists(&self.pending_generation_path(vm_id)).await
    }

    pub async fn select_generation(&self, vm_id: &str, generation: &str) -> Result<()> {
        check_vm_id(vm_id)?;
        if ulid::Ulid::from_string(generation).is_err() {
            bail!("invalid snapshot generation");
        }
        let dir = self.selected_snapshots_dir();
        fs::create_dir_all(&dir).await?;
        crate::durable::write_atomic(&dir.join(vm_id), generation.as_bytes()).map_err(Into::into)
    }

    async fn selected_generation(&self, vm_id: &str) -> Result<Option<String>> {
        match fs::read_to_string(self.selected_snapshots_dir().join(vm_id)).await {
            Ok(generation) => Ok(Some(generation)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).context("read selected snapshot generation"),
        }
    }

    fn removal_path(&self, vm_id: &str) -> PathBuf {
        self.removals_dir().join(vm_id)
    }

    pub fn checkpoint_dir(&self, vm_id: &str) -> PathBuf {
        self.root
            .join("checkpoints")
            .join(format!("{vm_id}-{}", ulid::Ulid::new()))
    }

    pub async fn write_checkpoint_metadata(
        &self,
        dir: &Path,
        kind: &str,
        manifest: &VmManifest,
    ) -> Result<()> {
        fs::write(dir.join(BACKEND_TAG_FILE), kind).await?;
        write_atomic_json(&dir.join(MANIFEST_FILE), manifest).await
    }

    /// Records which backend produced this snapshot. A Docker commit-based
    /// image, a `SmolVM` memory snapshot, and a Hypeman snapshot are mutually
    /// unreadable formats -- called right after a successful `suspend()` so
    /// a later `resume()` (possibly by a different backend, once this
    /// snapshot is restored elsewhere) can refuse before misinterpreting the
    /// directory contents instead of failing confusingly mid-restore.
    pub async fn write_backend_tag(&self, vm_id: &str, kind: &str) -> Result<()> {
        let path = self.snap_dir(vm_id).join(BACKEND_TAG_FILE);
        fs::write(&path, kind).await.context("write backend tag")
    }

    /// Refuses with a clear error if the snapshot was written by a different
    /// backend than `kind`. Missing metadata fails closed: without a format
    /// tag another worker cannot prove that its hypervisor can read the
    /// snapshot safely.
    pub async fn check_backend_tag(&self, vm_id: &str, kind: &str) -> Result<()> {
        check_backend_tag_at(&self.snap_dir(vm_id), vm_id, kind).await
    }

    /// Downloads a complete, committed snapshot and validates its format
    /// without creating a VM record or backend process. The control plane
    /// uses this before atomically transferring ownership to this worker.
    pub async fn prepare_claim(
        &self,
        vm_id: &str,
        generation: i64,
        snapshot_generation: &str,
        kind: &str,
    ) -> Result<()> {
        check_vm_id(vm_id)?;
        let dir = self.claim_dir(vm_id, generation);
        let staging = dir.with_extension("preparing");
        let _ = fs::remove_dir_all(&staging).await;
        fs::create_dir_all(&staging).await?;
        if !self
            .restore_generation_into(vm_id, snapshot_generation, &staging)
            .await?
        {
            bail!("no stored snapshot for {vm_id}");
        }
        check_backend_tag_at(&staging, vm_id, kind).await?;
        fs::write(staging.join(LOCAL_GENERATION_FILE), snapshot_generation).await?;
        let manifest = fs::read(staging.join(MANIFEST_FILE))
            .await
            .context("read prepared VM manifest")?;
        let manifest: VmManifest =
            serde_json::from_slice(&manifest).context("parse prepared VM manifest")?;
        if manifest.vm_id != vm_id {
            bail!(
                "prepared snapshot manifest names VM {}, expected {vm_id}",
                manifest.vm_id
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::rename(&staging, &dir).context("publish prepared VM snapshot")?;
        Ok(())
    }

    pub async fn has_claim(&self, vm_id: &str, generation: i64) -> Result<bool> {
        check_vm_id(vm_id)?;
        match fs::read_to_string(self.snap_dir(vm_id).join(CLAIM_TOKEN_FILE)).await {
            Ok(existing) => Ok(existing == generation.to_string()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error).context("read VM claim token"),
        }
    }

    pub async fn abandon_claim(&self, vm_id: &str, generation: i64) -> Result<bool> {
        check_vm_id(vm_id)?;
        if !self.has_claim(vm_id, generation).await? {
            return Ok(false);
        }
        self.remove_local_vm_state(vm_id).await?;
        Ok(true)
    }

    /// Promotes bytes prepared before the control-plane ownership CAS.
    pub async fn claim(&self, vm_id: &str, generation: i64) -> Result<VmManifest> {
        check_vm_id(vm_id)?;
        let owned = self.snap_dir(vm_id);
        if let Ok(existing) = fs::read_to_string(owned.join(CLAIM_TOKEN_FILE)).await {
            if existing == generation.to_string() {
                let bytes = fs::read(owned.join(MANIFEST_FILE))
                    .await
                    .context("read claimed VM manifest")?;
                return serde_json::from_slice(&bytes).context("parse claimed VM manifest");
            }
            bail!("VM {vm_id} was claimed with a different generation");
        }
        let prepared = self.claim_dir(vm_id, generation);
        let bytes = fs::read(prepared.join(MANIFEST_FILE))
            .await
            .context("read prepared VM manifest")?;
        let manifest: VmManifest =
            serde_json::from_slice(&bytes).context("parse prepared VM manifest")?;
        if manifest.vm_id != vm_id {
            bail!("prepared snapshot belongs to {}", manifest.vm_id);
        }
        fs::write(prepared.join(CLAIM_TOKEN_FILE), generation.to_string()).await?;
        if fs::try_exists(&owned).await? {
            bail!("VM {vm_id} already has local snapshot state without this claim token");
        }
        fs::create_dir_all(self.snapshots_dir())
            .await
            .context("create local snapshot directory")?;
        fs::rename(&prepared, &owned)
            .await
            .context("promote prepared snapshot")?;
        self.write_owned_manifest(&manifest).await?;
        Ok(manifest)
    }
}

mod io;
mod storage;
use io::*;

#[cfg(test)]
mod tests;
