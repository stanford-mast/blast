use super::*;

impl SnapshotStore {
    pub const fn has_shared_storage(&self) -> bool {
        self.shared_storage
    }

    /// Writes the manifest into the local snapshot, which lets this worker
    /// take the VM back after a restart.
    pub async fn write_manifest(&self, manifest: &VmManifest) -> Result<()> {
        check_vm_id(&manifest.vm_id)?;
        let dir = self.snap_dir(&manifest.vm_id);
        fs::create_dir_all(&dir)
            .await
            .context("create VM manifest directory")?;
        let path = dir.join(MANIFEST_FILE);
        let temporary = dir.join(format!(".{MANIFEST_FILE}.{}.tmp", ulid::Ulid::new()));
        let bytes = serde_json::to_vec(manifest)?;
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .await
            .context("create temporary VM manifest")?;
        if let Err(error) = async {
            file.write_all(&bytes).await?;
            file.sync_all().await?;
            atomic_replace(&temporary, &path).await?;
            Ok::<(), std::io::Error>(())
        }
        .await
        {
            let _ = fs::remove_file(&temporary).await;
            return Err(error).context("atomically write VM manifest");
        }
        sync_directory(&dir).await?;
        self.write_owned_manifest(manifest).await
    }

    pub(super) async fn write_owned_manifest(&self, manifest: &VmManifest) -> Result<()> {
        fs::create_dir_all(self.owned_dir()).await?;
        write_atomic_json(&self.owned_dir().join(&manifest.vm_id), manifest).await
    }

    pub async fn adopt_local_manifest(&self, vm_id: &str) -> Result<Option<VmManifest>> {
        check_vm_id(vm_id)?;
        if self.removal_pending(vm_id).await {
            return Ok(None);
        }
        let bytes = match fs::read(self.snap_dir(vm_id).join(MANIFEST_FILE)).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).context("read local VM manifest"),
        };
        let manifest: VmManifest =
            serde_json::from_slice(&bytes).context("parse local VM manifest")?;
        if manifest.vm_id != vm_id {
            bail!("local VM manifest {vm_id} names {}", manifest.vm_id);
        }
        self.write_owned_manifest(&manifest).await?;
        Ok(Some(manifest))
    }

    pub async fn adopt_all_local_manifests(&self) -> Result<()> {
        let mut entries = match fs::read_dir(self.snapshots_dir()).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error).context("read local snapshots"),
        };
        while let Some(entry) = entries.next_entry().await? {
            let Some(vm_id) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if is_safe_vm_id(&vm_id) {
                self.adopt_local_manifest(&vm_id).await?;
            }
        }
        Ok(())
    }

    /// Copies the local snapshot, manifest included, to durable storage,
    /// replacing any earlier copy.
    #[cfg(test)]
    pub async fn persist(&self, vm_id: &str) -> Result<()> {
        let generation = ulid::Ulid::new().to_string();
        self.persist_generation(vm_id, &generation).await?;
        self.select_generation(vm_id, &generation).await
    }

    pub async fn persist_generation(&self, vm_id: &str, generation: &str) -> Result<()> {
        check_vm_id(vm_id)?;
        let dir = self.snap_dir(vm_id);
        self.persist_from(vm_id, &dir, generation).await
    }

    pub async fn persist_checkpoint_generation(
        &self,
        vm_id: &str,
        dir: &Path,
        generation: &str,
    ) -> Result<()> {
        check_vm_id(vm_id)?;
        self.persist_from(vm_id, dir, generation).await
    }

    async fn persist_from(&self, vm_id: &str, dir: &Path, generation: &str) -> Result<()> {
        if ulid::Ulid::from_string(generation).is_err() {
            bail!("invalid snapshot generation");
        }
        let mut files = list_files(dir).await?;
        files.retain(|f| f != LOCAL_GENERATION_FILE);
        if !files.iter().any(|f| f == MANIFEST_FILE) {
            bail!("the snapshot of {vm_id} has no manifest");
        }
        if !files.iter().any(|f| f == BACKEND_TAG_FILE) {
            bail!("the snapshot of {vm_id} has no backend format tag");
        }
        let manifest: VmManifest = serde_json::from_slice(
            &fs::read(dir.join(MANIFEST_FILE))
                .await
                .context("read VM manifest before storing")?,
        )
        .context("parse VM manifest before storing")?;
        if manifest.vm_id != vm_id || manifest.backend_handle.is_some() {
            bail!("the snapshot of {vm_id} does not have a suspended VM manifest");
        }
        let generation = generation.to_owned();
        let mut checksums = BTreeMap::new();
        for rel in &files {
            checksums.insert(rel.clone(), file_sha256(&dir.join(rel)).await?);
        }
        match &self.storage {
            Storage::None => Ok(()),
            Storage::Dir(storage_dir) => {
                let root = storage_dir.join("vms").join(vm_id);
                let previous = self.selected_generation(vm_id).await?;
                let dest = root.join("generations").join(&generation);
                for rel in &files {
                    copy_file(&dir.join(rel), &dest.join(rel)).await?;
                }
                let commit = SnapshotCommit {
                    generation,
                    previous,
                    files: checksums,
                };
                write_atomic_json(&commit_path(&root, &commit.generation), &commit).await
            }
            Storage::Object { .. } => {
                let root = format!("vms/{vm_id}/");
                let previous = self.selected_generation(vm_id).await?;
                let prefix = format!("{root}generations/{generation}/");
                for rel in &files {
                    self.bucket()
                        .await?
                        .put_file(&format!("{prefix}{rel}"), &dir.join(rel))
                        .await?;
                }
                let commit = SnapshotCommit {
                    generation,
                    previous,
                    files: checksums,
                };
                let bytes = serde_json::to_vec(&commit)?;
                let bucket = self.bucket().await?;
                bucket
                    .put_bytes(
                        &format!("{root}{COMMITS_DIR}/{}.json", commit.generation),
                        bytes,
                    )
                    .await
            }
        }
    }

    /// The manifest of a snapshot this worker holds locally or can restore
    /// from storage; `None` when neither has a complete one. An error means
    /// storage could not answer, not that the VM is unknown.
    #[cfg(test)]
    pub async fn find_manifest(&self, vm_id: &str) -> Result<Option<VmManifest>> {
        if !is_safe_vm_id(vm_id) {
            return Ok(None);
        }
        if let Storage::Object { bucket, .. } = &self.storage {
            if bucket.read().await.is_none() {
                if let Some(manifest) = self.find_local_manifest(vm_id).await? {
                    return Ok(Some(manifest));
                }
                bail!("the control plane has not answered a heartbeat yet");
            }
        }
        let stored = self
            .selected_generation(vm_id)
            .await?
            .map(|generation| SnapshotCommit {
                generation,
                previous: None,
                files: BTreeMap::new(),
            });
        let bytes = if let Some(commit) = stored {
            let path = format!("generations/{}/{MANIFEST_FILE}", commit.generation);
            match &self.storage {
                Storage::Dir(root) => {
                    Some(fs::read(root.join("vms").join(vm_id).join(path)).await?)
                }
                Storage::Object { .. } => {
                    self.object_manifest(vm_id).await?.map(|(_, bytes)| bytes)
                }
                Storage::None => unreachable!(),
            }
        } else {
            match fs::read(self.snap_dir(vm_id).join(MANIFEST_FILE)).await {
                Ok(bytes) => Some(bytes),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => return Err(e).context("read VM manifest"),
            }
        };
        bytes
            .map(|b| serde_json::from_slice(&b).context("parse VM manifest"))
            .transpose()
    }

    /// Host-local identity is considered separately from durable checkpoint
    /// state so a daemon-backed VM that survived this process can be
    /// reattached instead of restored beside itself.
    #[cfg(test)]
    pub async fn find_local_manifest(&self, vm_id: &str) -> Result<Option<VmManifest>> {
        if !is_safe_vm_id(vm_id) {
            return Ok(None);
        }
        match fs::read(self.snap_dir(vm_id).join(MANIFEST_FILE)).await {
            Ok(bytes) => Ok(Some(
                serde_json::from_slice(&bytes).context("parse local VM manifest")?,
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).context("read local VM manifest"),
        }
    }

    pub async fn owned_manifests(&self) -> Result<Vec<VmManifest>> {
        let mut manifests = Vec::new();
        let mut entries = match fs::read_dir(self.owned_dir()).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(manifests),
            Err(error) => return Err(error).context("read local snapshots"),
        };
        while let Some(entry) = entries.next_entry().await? {
            let Some(vm_id) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if self.removal_pending(&vm_id).await {
                continue;
            }
            let bytes = fs::read(entry.path()).await?;
            let manifest: VmManifest =
                serde_json::from_slice(&bytes).context("parse owned VM record")?;
            if manifest.vm_id != vm_id {
                bail!("owned VM record {vm_id} names {}", manifest.vm_id);
            }
            manifests.push(manifest);
        }
        Ok(manifests)
    }

    /// Makes sure the VM's snapshot is on local disk, restoring it from
    /// storage when it is not. Errors when no complete copy exists anywhere.
    pub async fn ensure_local(&self, vm_id: &str) -> Result<PathBuf> {
        self.ensure_local_mode(vm_id).await
    }

    async fn ensure_local_mode(&self, vm_id: &str) -> Result<PathBuf> {
        check_vm_id(vm_id)?;
        let dir = self.snap_dir(vm_id);
        let staging = self
            .snapshots_dir()
            .join(format!(".restore-{vm_id}-{}", ulid::Ulid::new()));
        let restored = match self.selected_generation(vm_id).await? {
            Some(generation) if self.local_generation(vm_id).await? == Some(generation.clone()) => {
                return Ok(dir);
            }
            Some(generation) => {
                let restored = self
                    .restore_generation_into(vm_id, &generation, &staging)
                    .await;
                if matches!(restored, Ok(true)) {
                    fs::write(staging.join(LOCAL_GENERATION_FILE), &generation).await?;
                }
                restored
            }
            None if self.shared_storage => {
                bail!("no control-plane-selected snapshot generation for {vm_id}")
            }
            None => self.restore_into(vm_id, &staging).await,
        };
        match restored {
            Ok(true) => {
                let old = self
                    .snapshots_dir()
                    .join(format!(".old-{vm_id}-{}", ulid::Ulid::new()));
                let had_old = fs::try_exists(&dir).await?;
                if had_old {
                    fs::rename(&dir, &old)
                        .await
                        .context("park stale local snapshot")?;
                }
                if let Err(error) = fs::rename(&staging, &dir).await {
                    if had_old {
                        let _ = fs::rename(&old, &dir).await;
                    }
                    return Err(error).context("move restored snapshot");
                }
                let _ = fs::remove_dir_all(old).await;
                tracing::info!(vm_id, "restored snapshot from storage");
                Ok(dir)
            }
            Ok(false) => {
                let _ = fs::remove_dir_all(&staging).await;
                if fs::try_exists(&dir).await? {
                    Ok(dir)
                } else {
                    bail!("no snapshot for {vm_id} locally or in storage")
                }
            }
            Err(e) => {
                let _ = fs::remove_dir_all(&staging).await;
                Err(e.context(format!("restore snapshot for {vm_id}")))
            }
        }
    }

    async fn local_generation(&self, vm_id: &str) -> Result<Option<String>> {
        match fs::read_to_string(self.snap_dir(vm_id).join(LOCAL_GENERATION_FILE)).await {
            Ok(generation) => Ok(Some(generation)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).context("read local snapshot generation"),
        }
    }

    /// Called before the local snapshot is rewritten, so it is no longer
    /// taken for the stored generation it was restored from.
    pub async fn invalidate_local_generation(&self, vm_id: &str) -> Result<()> {
        check_vm_id(vm_id)?;
        remove_file_if_exists(&self.snap_dir(vm_id).join(LOCAL_GENERATION_FILE)).await
    }

    /// The snapshot a VM would resume from, locally or in storage; `None`
    /// when it has none.
    pub async fn last_snapshot(&self, vm_id: &str) -> Result<Option<LastSnapshot>> {
        check_vm_id(vm_id)?;
        let dir = self.snap_dir(vm_id);
        let local = fs::try_exists(dir.join(BACKEND_TAG_FILE))
            .await
            .context("inspect local snapshot")?
            .then_some(dir);
        let stored = self.selected_generation(vm_id).await?.is_some();
        Ok((local.is_some() || stored).then_some(LastSnapshot { local, stored }))
    }

    /// Hard-links the local snapshot into a transient directory, which keeps
    /// it readable after the VM's own directory is replaced or removed.
    /// Snapshot files are replaced, never rewritten in place.
    pub async fn pin_local(&self, vm_id: &str) -> Result<TransientSnapshot> {
        let source = self.snap_dir(vm_id);
        let pinned = self.transient_snapshot(vm_id).await?;
        for rel in list_files(&source).await? {
            let (from, to) = (source.join(&rel), pinned.path().join(&rel));
            if let Some(parent) = to.parent() {
                fs::create_dir_all(parent).await?;
            }
            if fs::hard_link(&from, &to).await.is_err() {
                copy_file(&from, &to).await?;
            }
        }
        Ok(pinned)
    }

    /// Copies the stored snapshot into `dest`; `false` when storage holds no
    /// complete one.
    pub(super) async fn restore_generation_into(
        &self,
        vm_id: &str,
        generation: &str,
        dest: &Path,
    ) -> Result<bool> {
        if ulid::Ulid::from_string(generation).is_err() {
            bail!("invalid snapshot generation");
        }
        let commit = match &self.storage {
            Storage::None => return Ok(false),
            Storage::Dir(storage_dir) => {
                let root = storage_dir.join("vms").join(vm_id);
                let bytes = match fs::read(commit_path(&root, generation)).await {
                    Ok(bytes) => bytes,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
                    Err(error) => return Err(error).context("read committed snapshot generation"),
                };
                let commit: SnapshotCommit = serde_json::from_slice(&bytes)
                    .context("parse committed snapshot generation")?;
                validate_commit_shape(&commit)?;
                if commit.generation != generation {
                    bail!("snapshot commit names a different generation");
                }
                let source = root.join("generations").join(generation);
                for (rel, expected) in &commit.files {
                    copy_file(&source.join(rel), &dest.join(rel)).await?;
                    verify_sha256(&dest.join(rel), expected).await?;
                }
                return Ok(true);
            }
            Storage::Object { .. } => {
                let root = format!("vms/{vm_id}/");
                let key = format!("{root}{COMMITS_DIR}/{generation}.json");
                let Some(bytes) = self.bucket().await?.get_bytes(&key).await? else {
                    return Ok(false);
                };
                serde_json::from_slice(&bytes).context("parse committed snapshot generation")?
            }
        };
        validate_commit_shape(&commit)?;
        if commit.generation != generation {
            bail!("snapshot commit names a different generation");
        }
        let bucket = self.bucket().await?;
        let prefix = format!("vms/{vm_id}/generations/{generation}/");
        for (rel, expected) in &commit.files {
            bucket
                .get_file(&format!("{prefix}{rel}"), &dest.join(rel))
                .await?;
            verify_sha256(&dest.join(rel), expected).await?;
        }
        Ok(true)
    }

    async fn restore_into(&self, vm_id: &str, dest: &Path) -> Result<bool> {
        match &self.storage {
            Storage::None => Ok(false),
            Storage::Dir(storage_dir) => {
                let root = storage_dir.join("vms").join(vm_id);
                let Some(commit) = resolve_dir_commit(&root).await? else {
                    return Ok(false);
                };
                let src = root.join("generations").join(&commit.generation);
                for (rel, expected) in &commit.files {
                    copy_file(&src.join(rel), &dest.join(rel)).await?;
                    verify_sha256(&dest.join(rel), expected).await?;
                }
                Ok(true)
            }
            Storage::Object { .. } => {
                let candidates = self.object_commit_candidates(vm_id).await?;
                if candidates.is_empty() {
                    return Ok(false);
                }
                for commit in candidates {
                    if validate_commit_shape(&commit).is_err() {
                        continue;
                    }
                    let prefix = format!("vms/{vm_id}/generations/{}/", commit.generation);
                    let mut valid = true;
                    for (rel, expected) in &commit.files {
                        let destination = dest.join(rel);
                        if !self
                            .bucket()
                            .await?
                            .get_file(&format!("{prefix}{rel}"), &destination)
                            .await?
                            || verify_sha256(&destination, expected).await.is_err()
                        {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        return Ok(true);
                    }
                    let _ = fs::remove_dir_all(dest).await;
                    fs::create_dir_all(dest).await?;
                }
                bail!("no valid committed snapshot generation for {vm_id}")
            }
        }
    }

    pub async fn forget(&self, vm_id: &str) -> Result<()> {
        check_vm_id(vm_id)?;
        let removals = self.removals_dir();
        fs::create_dir_all(&removals).await?;
        let tombstone = fs::File::create(self.removal_path(vm_id)).await?;
        tombstone.sync_all().await?;
        sync_directory(&removals).await?;
        self.remove_local_vm_state(vm_id).await
    }

    /// Drops a VM's worker-local checkpoint after its durable copy commits.
    /// Only shared storage may do this: local-only storage is part of this
    /// worker's disk allocation and remains the sole resumable copy.
    pub async fn release_local_snapshot(&self, vm_id: &str) -> Result<()> {
        if !self.shared_storage {
            bail!("cannot release a checkpoint without shared storage");
        }
        self.invalidate_local_generation(vm_id).await?;
        match fs::remove_dir_all(self.snap_dir(vm_id)).await {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error.into()),
            _ => Ok(()),
        }
    }

    pub async fn discard_local_vm_state(&self, vm_id: &str) -> Result<()> {
        check_vm_id(vm_id)?;
        self.remove_local_vm_state(vm_id).await
    }

    /// Removes every local snapshot, including those of VMs this worker
    /// has not taken back since a restart.
    pub async fn discard_worker_state(&self) -> Result<()> {
        for directory in [
            self.snapshots_dir(),
            self.root.join("claims"),
            self.owned_dir(),
            self.selected_snapshots_dir(),
            self.root.join("pending-snapshots"),
            self.removals_dir(),
            self.local_discards_dir(),
        ] {
            match fs::remove_dir_all(directory).await {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                    return Err(error.into())
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub async fn remove_stored(&self, vm_id: &str) {
        if !self.shared_storage {
            if let Storage::Dir(storage_dir) = &self.storage {
                let durable = storage_dir.join("vms").join(vm_id);
                if let Err(error) = remove_dir_if_exists(&durable).await {
                    tracing::warn!(vm_id, err = %format!("{error:#}"), "removing local durable VM state failed; will retry");
                    return;
                }
            }
        }
        if let Err(error) = self.remove_local_vm_state(vm_id).await {
            tracing::warn!(vm_id, err = %format!("{error:#}"), "removing local VM state failed; will retry");
            return;
        }
        let _ = fs::remove_file(self.removal_path(vm_id)).await;
    }

    pub async fn retry_removals(&self) {
        let Ok(mut entries) = fs::read_dir(self.removals_dir()).await else {
            return;
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            if let Some(vm_id) = entry.file_name().to_str().filter(|id| is_safe_vm_id(id)) {
                self.remove_stored(vm_id).await;
            }
        }
    }

    pub async fn pending_removals(&self) -> Result<Vec<String>> {
        let mut pending = Vec::new();
        let mut entries = match fs::read_dir(self.removals_dir()).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(pending),
            Err(error) => return Err(error).context("read snapshot removals"),
        };
        while let Some(entry) = entries.next_entry().await? {
            if let Some(vm_id) = entry.file_name().to_str().filter(|id| is_safe_vm_id(id)) {
                pending.push(vm_id.to_owned());
            }
        }
        pending.sort();
        Ok(pending)
    }

    pub async fn removal_pending(&self, vm_id: &str) -> bool {
        fs::try_exists(self.removal_path(vm_id))
            .await
            .unwrap_or(true)
    }

    pub(super) async fn remove_local_vm_state(&self, vm_id: &str) -> Result<()> {
        remove_dir_if_exists(&self.snap_dir(vm_id)).await?;
        remove_dir_if_exists(&self.root.join("claims").join(vm_id)).await?;
        match fs::remove_file(self.owned_dir().join(vm_id)).await {
            Ok(()) => sync_directory(&self.owned_dir()).await?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("remove owned VM record"),
        }
        remove_file_if_exists(&self.selected_snapshots_dir().join(vm_id)).await?;
        remove_file_if_exists(&self.pending_generation_path(vm_id)).await?;
        Ok(())
    }

    async fn bucket(&self) -> Result<s3::Bucket> {
        let Storage::Object { bucket, .. } = &self.storage else {
            bail!("snapshots are not kept in object storage");
        };
        bucket
            .read()
            .await
            .as_deref()
            .cloned()
            .context("the control plane has not answered a heartbeat yet")
    }

    #[cfg(test)]
    async fn object_manifest(&self, vm_id: &str) -> Result<Option<(SnapshotCommit, Vec<u8>)>> {
        use sha2::{Digest, Sha256};

        let candidates = self.object_commit_candidates(vm_id).await?;
        if candidates.is_empty() {
            return Ok(None);
        }
        let bucket = self.bucket().await?;
        let root = format!("vms/{vm_id}/");
        for commit in candidates {
            if validate_commit_shape(&commit).is_err() {
                continue;
            }
            let key = format!("{root}generations/{}/{MANIFEST_FILE}", commit.generation);
            let Some(bytes) = bucket.get_bytes(&key).await? else {
                continue;
            };
            let expected = &commit.files[MANIFEST_FILE];
            if hex::encode(Sha256::digest(&bytes)) == *expected {
                return Ok(Some((commit, bytes)));
            }
        }
        bail!("no valid committed snapshot manifest for {vm_id}")
    }

    async fn object_commit_candidates(&self, vm_id: &str) -> Result<Vec<SnapshotCommit>> {
        let bucket = self.bucket().await?;
        let root = format!("vms/{vm_id}/");
        let head = bucket.get_bytes(&format!("{root}{COMMIT_FILE}")).await?;
        let history_prefix = format!("{root}{COMMITS_DIR}/");
        let history = bucket.list(&history_prefix).await?;
        if head.is_none() && history.is_empty() {
            return Ok(Vec::new());
        }

        let mut candidates = Vec::new();
        if let Some(bytes) = head {
            if let Ok(commit) = serde_json::from_slice(&bytes) {
                candidates.push(commit);
            }
        }
        let mut history = history;
        history.sort_by(|a, b| b.cmp(a));
        for rel in history {
            if Path::new(&rel).extension() != Some(std::ffi::OsStr::new("json"))
                || rel.contains('/')
            {
                continue;
            }
            let Some(bytes) = bucket.get_bytes(&format!("{history_prefix}{rel}")).await? else {
                continue;
            };
            if let Ok(commit) = serde_json::from_slice(&bytes) {
                candidates.push(commit);
            }
        }
        deduplicate_commits(&mut candidates);
        Ok(candidates)
    }
}
