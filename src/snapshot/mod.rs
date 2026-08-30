use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use reqwest::Client;
use sha2::{Digest, Sha256};
use tokio::fs;

pub struct SnapshotStore {
    root: PathBuf,
    client: Client,
}

impl SnapshotStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            client: Client::new(),
        }
    }

    #[allow(dead_code)]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Local path for a VM's snapshot directory.
    pub fn snap_dir(&self, vm_id: &str) -> PathBuf {
        self.root.join("snapshots").join(vm_id)
    }

    pub async fn ensure_dir(&self, vm_id: &str) -> Result<PathBuf> {
        let dir = self.snap_dir(vm_id);
        fs::create_dir_all(&dir).await.context("create snap dir")?;
        Ok(dir)
    }

    /// Filename for the backend-kind marker `write_backend_marker`/
    /// `check_backend_marker` share. Lives inside `snap_dir` itself so it
    /// travels with the snapshot through `upload_via_presigned`/
    /// `download_via_presigned` -- the same round trip that would carry a
    /// snapshot to wherever it's next resumed.
    const BACKEND_MARKER_FILE: &'static str = ".blast-backend";

    /// Records which backend produced this snapshot. A Docker commit-based
    /// image, a SmolVM memory snapshot, and a Hypeman snapshot are mutually
    /// unreadable formats -- called right after a successful `suspend()` so
    /// a later `resume()` (possibly by a different backend, if this snapshot
    /// is ever synced elsewhere) can refuse before misinterpreting the
    /// directory contents instead of failing confusingly mid-restore.
    pub async fn write_backend_marker(&self, vm_id: &str, kind: &str) -> Result<()> {
        let path = self.snap_dir(vm_id).join(Self::BACKEND_MARKER_FILE);
        fs::write(&path, kind).await.context("write backend marker")
    }

    /// Refuses with a clear error if `snap_dir` was written by a different
    /// backend than `kind`. A snapshot with no marker at all predates this
    /// check (or was written by a build that doesn't have it yet) and is let
    /// through -- absence isn't evidence of a mismatch, and failing every
    /// pre-existing snapshot would be worse than the corruption this guards
    /// against actually happening.
    pub async fn check_backend_marker(&self, vm_id: &str, kind: &str) -> Result<()> {
        let path = self.snap_dir(vm_id).join(Self::BACKEND_MARKER_FILE);
        let recorded = match fs::read_to_string(&path).await {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).context("read backend marker"),
        };
        if recorded != kind {
            anyhow::bail!(
                "snapshot for {vm_id} was written by backend '{recorded}', \
                 but this worker runs backend '{kind}' -- refusing to resume \
                 a snapshot this backend cannot read"
            );
        }
        Ok(())
    }

    /// Upload every file in `snap_dir` to the given presigned PUT URLs
    /// obtained from the control plane. Returns content hashes.
    ///
    /// `upload_url_fn(blob_key)` is called async to resolve the presigned PUT
    /// URL for each blob, typically a call to the router's storage/upload-url
    /// endpoint which in turn generates and returns an S3 presigned URL.
    pub async fn upload_via_presigned<F, Fut>(
        &self,
        snap_dir: &Path,
        upload_url_fn: F,
    ) -> Result<Vec<String>>
    where
        F: Fn(String) -> Fut,
        Fut: std::future::Future<Output = Result<String>>,
    {
        let mut hashes = Vec::new();
        let mut entries = fs::read_dir(snap_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let bytes = fs::read(&path).await?;
            let hash = hex::encode(Sha256::digest(&bytes));
            let url = upload_url_fn(hash.clone()).await.context("resolve upload URL")?;
            self.client
                .put(&url)
                .body(bytes)
                .send()
                .await
                .context("snapshot upload")?
                .error_for_status()
                .context("snapshot upload status")?;
            hashes.push(hash);
        }
        Ok(hashes)
    }

    /// Download blob identified by `hash` from a presigned GET URL into `dest`.
    #[allow(dead_code)]
    pub async fn download_via_presigned(&self, url: &str, dest: &Path) -> Result<()> {
        let bytes = self
            .client
            .get(url)
            .send()
            .await
            .context("snapshot download")?
            .error_for_status()
            .context("snapshot download status")?
            .bytes()
            .await?;
        fs::write(dest, &bytes).await.context("write snapshot file")?;
        Ok(())
    }
}
