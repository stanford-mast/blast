use super::*;

pub(super) fn commit_path(root: &Path, generation: &str) -> PathBuf {
    root.join(COMMITS_DIR).join(format!("{generation}.json"))
}

pub(super) async fn resolve_dir_commit(root: &Path) -> Result<Option<SnapshotCommit>> {
    let head = match fs::read(root.join(COMMIT_FILE)).await {
        Ok(bytes) => Some(bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).context("read snapshot commit"),
    };
    let history_dir = root.join(COMMITS_DIR);
    let mut history = match fs::read_dir(&history_dir).await {
        Ok(entries) => Some(entries),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).context("read snapshot commit history"),
    };
    if head.is_none() && history.is_none() {
        return Ok(None);
    }

    let mut candidates = Vec::new();
    if let Some(bytes) = head {
        if let Ok(commit) = serde_json::from_slice(&bytes) {
            candidates.push(commit);
        }
    }
    let mut paths = Vec::new();
    if let Some(entries) = history.as_mut() {
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                paths.push(entry.path());
            }
        }
    }
    paths.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    for path in paths {
        let Ok(bytes) = fs::read(path).await else {
            continue;
        };
        if let Ok(commit) = serde_json::from_slice(&bytes) {
            candidates.push(commit);
        }
    }
    deduplicate_commits(&mut candidates);
    for commit in candidates {
        if validate_commit_shape(&commit).is_err() {
            continue;
        }
        let generation = root.join("generations").join(&commit.generation);
        let mut valid = true;
        for (rel, expected) in &commit.files {
            let path = generation.join(rel);
            let metadata = match fs::symlink_metadata(&path).await {
                Ok(metadata) if metadata.file_type().is_file() => metadata,
                _ => {
                    valid = false;
                    break;
                }
            };
            if metadata.file_type().is_symlink()
                || file_sha256(&path).await.ok().as_deref() != Some(expected)
            {
                valid = false;
                break;
            }
        }
        if valid {
            return Ok(Some(commit));
        }
    }
    bail!(
        "no valid committed snapshot generation in {}",
        root.display()
    )
}

pub(super) fn deduplicate_commits(commits: &mut Vec<SnapshotCommit>) {
    let mut seen = HashSet::new();
    commits.retain(|commit| seen.insert(commit.generation.clone()));
}

pub(super) fn validate_commit_shape(commit: &SnapshotCommit) -> Result<()> {
    if ulid::Ulid::from_string(&commit.generation).is_err() {
        bail!("invalid snapshot generation");
    }
    if !commit.files.contains_key(MANIFEST_FILE) || !commit.files.contains_key(BACKEND_TAG_FILE) {
        bail!("snapshot commit lacks required metadata");
    }
    for (rel, checksum) in &commit.files {
        if !is_safe_relative(rel)
            || checksum.len() != 64
            || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("invalid snapshot commit entry {rel:?}");
        }
    }
    Ok(())
}

pub(super) async fn write_atomic_json(path: &Path, value: &(impl Serialize + Sync)) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    let parent = path.parent().context("snapshot commit has no parent")?;
    fs::create_dir_all(parent).await?;
    let temporary = parent.join(format!(".{COMMIT_FILE}.{}.tmp", ulid::Ulid::new()));
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .await?;
    file.write_all(&bytes).await?;
    file.sync_all().await?;
    atomic_replace(&temporary, path).await?;
    sync_directory(parent).await
}

pub(super) async fn file_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use tokio::io::AsyncReadExt;
    let mut file = fs::File::open(path).await?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

pub(super) async fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let actual = file_sha256(path).await?;
    if actual != expected {
        bail!("snapshot checksum mismatch for {}", path.display());
    }
    Ok(())
}

pub(super) async fn check_backend_tag_at(dir: &Path, vm_id: &str, kind: &str) -> Result<()> {
    let path = dir.join(BACKEND_TAG_FILE);
    let recorded = match fs::read_to_string(&path).await {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!("snapshot for {vm_id} has no backend format tag; refusing to resume it")
        }
        Err(e) => return Err(e).context("read backend tag"),
    };
    if recorded != kind {
        bail!(
            "snapshot for {vm_id} was written by backend '{recorded}', but this worker runs \
             backend '{kind}' -- refusing to resume a snapshot this backend cannot read"
        );
    }
    Ok(())
}

#[cfg(not(windows))]
pub(super) async fn atomic_replace(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::rename(source, destination).await
}

#[cfg(windows)]
pub(super) async fn atomic_replace(source: &Path, destination: &Path) -> std::io::Result<()> {
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
    tokio::task::spawn_blocking(move || {
        // SAFETY: both pointers name owned, NUL-terminated UTF-16 buffers
        // that live for the duration of the call.
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
    })
    .await
    .map_err(std::io::Error::other)?
}

#[cfg(not(windows))]
pub(super) async fn sync_directory(dir: &Path) -> Result<()> {
    let directory = fs::File::open(dir)
        .await
        .context("open VM manifest directory")?;
    directory
        .sync_all()
        .await
        .context("sync VM manifest directory")
}

#[cfg(windows)]
pub(super) async fn sync_directory(_: &Path) -> Result<()> {
    // MoveFileExW with WRITE_THROUGH flushes the replacement before return.
    Ok(())
}

/// Every regular file under `dir`, as `/`-separated paths relative to it.
pub(super) async fn list_files(dir: &Path) -> Result<Vec<String>> {
    let mut files = Vec::new();
    let mut pending = vec![PathBuf::new()];
    while let Some(rel) = pending.pop() {
        let mut entries = fs::read_dir(dir.join(&rel))
            .await
            .with_context(|| format!("read {}", dir.join(&rel).display()))?;
        while let Some(entry) = entries.next_entry().await? {
            let path = rel.join(entry.file_name());
            let kind = entry.file_type().await?;
            if kind.is_dir() {
                pending.push(path);
            } else if kind.is_file() {
                let text = path.to_str().context("snapshot file name is not UTF-8")?;
                files.push(text.replace(std::path::MAIN_SEPARATOR, "/"));
            } else {
                bail!(
                    "refusing non-regular snapshot entry {}",
                    dir.join(path).display()
                );
            }
        }
    }
    files.sort();
    Ok(files)
}

/// A stored file name that stays inside the directory it is restored into.
pub(super) fn is_safe_relative(rel: &str) -> bool {
    !rel.is_empty()
        && !rel.starts_with('/')
        && rel
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

pub(super) async fn copy_file(src: &Path, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::copy(src, dest)
        .await
        .with_context(|| format!("copy {} to {}", src.display(), dest.display()))?;
    fs::OpenOptions::new()
        .write(true)
        .open(dest)
        .await
        .with_context(|| format!("open copied snapshot {}", dest.display()))?
        .sync_all()
        .await
        .with_context(|| format!("sync copied snapshot {}", dest.display()))?;
    Ok(())
}

pub(super) async fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path).await {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            Err(e).with_context(|| format!("remove {}", path.display()))
        }
        _ => Ok(()),
    }
}

pub(super) async fn remove_dir_if_exists(path: &Path) -> Result<()> {
    match fs::remove_dir_all(path).await {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
            Err(error).with_context(|| format!("remove {}", path.display()))
        }
        _ => Ok(()),
    }
}
