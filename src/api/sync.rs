//! `POST /vms/{id}/sync`: inline file reads and writes.
//!
//! Writes follow the API's chunked protocol: each entry carries the byte range
//! `[start, end)` of a `size`-byte file, chunks of one file share an
//! `upload_id`, and the destination is replaced only once one upload has
//! covered `[0, size)`. Chunks are staged on the host under
//! `{data_dir}/sync-staging` and the finished file is handed to the backend's
//! `write_file`, which streams it in rather than through a command line.

use std::{
    collections::HashMap,
    io::SeekFrom,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    extract::{Path, State},
    Json,
};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use crate::{
    backend::{shell_quote, Ended, Exec, Stdout, VmHandle},
    worker::records::Activity,
};

use super::{
    error::ApiError,
    handlers::{begin_mutation, ensure_running_handle, find_vm, lock_vm, AppState},
    types::{ByteRange, SyncFileError, SyncRequest, SyncResponse, SyncWrite, SyncWriteResult},
};

/// An upload idle this long is abandoned: its staging file is removed.
const UPLOAD_TTL: Duration = Duration::from_secs(15 * 60);

/// Exit status the read command uses for "no such file".
const MISSING_FILE_EXIT: i32 = 44;

/// The largest file a read returns inline. The API hands anything larger
/// out through a presigned download, which this worker does not offer.
const INLINE_READ_LIMIT: u64 = 16 * 1024 * 1024;

/// The path as the guest is asked for it: absolute, with no `.` or `..`.
fn sanitize_path(path: &str) -> Result<String, ApiError> {
    use std::path::Component;
    let refuse = |message: String| ApiError::bad_request_field("path", message);
    if path.is_empty() {
        return Err(refuse("path is required".into()));
    }
    if !path.starts_with('/') {
        return Err(refuse(format!("path must be absolute: {path}")));
    }
    let mut parts = Vec::new();
    for component in std::path::Path::new(path).components() {
        match component {
            Component::RootDir => {}
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            _ => return Err(refuse(format!("path escapes sandbox: {path}"))),
        }
    }
    if parts.is_empty() {
        return Err(refuse("path must name a file".into()));
    }
    Ok(format!("/{}", parts.join("/")))
}

/// In-flight chunked uploads, keyed by `(vm_id, upload_id)`.
pub type Uploads = Arc<Mutex<HashMap<(String, String), Upload>>>;

pub struct Upload {
    staging: PathBuf,
    path: String,
    size: u64,
    /// The byte ranges `[start, end)` received, each with the sha256 of its
    /// bytes, so the same chunk sent again is recognised.
    chunks: Vec<((u64, u64), [u8; 32])>,
    last_activity: Instant,
}

pub async fn post_sync(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
    Json(req): Json<SyncRequest>,
) -> Result<Json<SyncResponse>, ApiError> {
    // Reads may resume a suspended VM, so every sync operation is a mutation
    // for lifecycle-barrier purposes.
    let _mutation = begin_mutation(&s).await?;
    let _vm = lock_vm(&s, &vm_id).await;
    let req = match req {
        SyncRequest::Manifest { path } => {
            sanitize_path(&path)?;
            return Err(ApiError::unsupported(
                Some("sync_manifest"),
                "sync manifests are not supported",
            ));
        }
        SyncRequest::Write { mut writes } => {
            if writes.is_empty() {
                return Err(ApiError::bad_request("`writes` must not be empty"));
            }
            for write in &mut writes {
                write.path = sanitize_path(&write.path)?;
            }
            if writes
                .iter()
                .any(|w| w.presigned == Some(true) || w.content.is_none())
            {
                return Err(presigned_refusal(
                    "presigned uploads are not supported; send `content` inline",
                ));
            }
            Parsed::Write(chunks(writes)?)
        }
        SyncRequest::Read { path } => Parsed::Read(sanitize_path(&path)?),
    };
    find_vm(&s, &vm_id).await?;

    let result = async {
        let handle = ensure_running_handle(&s, &vm_id).await?;
        match req {
            Parsed::Read(path) => read(&s, &handle, path).await,
            Parsed::Write(chunks) => {
                let mut results = Vec::with_capacity(chunks.len());
                for chunk in chunks {
                    results.push(write(&s, &vm_id, &handle, chunk).await);
                }
                let ok = results.iter().all(|r| r.error.is_none());
                Ok(SyncResponse::Write { ok, results })
            }
        }
    }
    .await;
    if let Err(error) = s.records.activity(Activity::operation(
        "sync",
        if result.is_ok() {
            "completed"
        } else {
            "failed"
        },
        Some(vm_id),
    )) {
        tracing::warn!(err = %error, "persist sync activity failed");
    }
    result.map(Json)
}

fn presigned_refusal(message: &str) -> ApiError {
    ApiError::unsupported(Some("presigned_sync"), message)
}

/// Reads the file out through a host staging file, so its size is bounded
/// by the inline limit and not by what one command may print.
async fn read(s: &AppState, handle: &VmHandle, path: String) -> Result<SyncResponse, ApiError> {
    let staging = s
        .config
        .data_dir
        .join("sync-staging")
        .join(format!("read-{}", ulid::Ulid::new()));
    if let Some(dir) = staging.parent() {
        tokio::fs::create_dir_all(dir)
            .await
            .map_err(|e| ApiError::internal(format!("stage read: {e}")))?;
    }
    let read = read_staged(s, handle, &path, &staging).await;
    tokio::fs::remove_file(&staging).await.ok();
    let bytes = read?;
    let size = bytes.len() as u64;
    let (content, encoding) = match String::from_utf8(bytes) {
        Ok(text) => (text, "utf8"),
        Err(e) => (B64.encode(e.as_bytes()), "base64"),
    };
    Ok(SyncResponse::Read {
        ok: true,
        path,
        size,
        content,
        encoding,
    })
}

/// Bytes one guest command hands back: a backend caps what a command may
/// print, so a file is read in pieces under that cap.
const READ_CHUNK: u64 = 4 * 1024 * 1024;

async fn read_staged(
    s: &AppState,
    handle: &VmHandle,
    path: &str,
    staging: &std::path::Path,
) -> Result<Vec<u8>, ApiError> {
    let quoted = shell_quote(path);
    let mut bytes = Vec::new();
    // One byte past the limit tells a file at the limit from a larger one.
    while bytes.len() as u64 <= INLINE_READ_LIMIT {
        let command = format!(
            "if [ -e {quoted} ]; then tail -c +{} -- {quoted} | head -c {READ_CHUNK}; \
             else exit {MISSING_FILE_EXIT}; fi",
            bytes.len() + 1
        );
        let out = s
            .backend
            .run(handle, Exec::internal(&command, Stdout::File(staging)))
            .await?;
        match out.ended {
            Ended::Exited(0) => {}
            Ended::Exited(MISSING_FILE_EXIT) => return Err(ApiError::not_found("file")),
            ended => {
                return Err(ApiError::internal(format!(
                    "read {path}: {ended:?}: {}",
                    String::from_utf8_lossy(&out.stderr.bytes).trim()
                )))
            }
        }
        let chunk = tokio::fs::read(staging)
            .await
            .map_err(|e| ApiError::internal(format!("read staged {path}: {e}")))?;
        let last = (chunk.len() as u64) < READ_CHUNK;
        bytes.extend(chunk);
        if last {
            break;
        }
    }
    if bytes.len() as u64 > INLINE_READ_LIMIT {
        return Err(presigned_refusal(
            "files larger than 16 MiB are read through presigned downloads, which are not supported",
        ));
    }
    Ok(bytes)
}

/// A sync request that passed the checks made before anything is written.
enum Parsed {
    Read(String),
    Write(Vec<Chunk>),
}

/// One inline write: a chunk of an upload, or a whole file.
struct Chunk {
    path: String,
    size: u64,
    /// Empty for a whole file sent on its own.
    upload_id: String,
    bytes: Vec<u8>,
    start: u64,
    end: u64,
    sha256: Option<String>,
    secret: bool,
}

/// The largest file a write may declare.
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// The most one chunk carries.
const MAX_CHUNK_BYTES: usize = 5 * 1024 * 1024;
/// The most all the chunks of one request carry.
const MAX_REQUEST_BYTES: usize = 20 * 1024 * 1024;

/// Decodes and checks every write of a request. One bad write refuses the
/// whole request, before any of it is written.
fn chunks(writes: Vec<SyncWrite>) -> Result<Vec<Chunk>, ApiError> {
    let too_large = || ApiError::payload_too_large("the request is larger than a sync accepts");
    let mut total = 0usize;
    let mut parsed = Vec::with_capacity(writes.len());
    for (index, w) in writes.into_iter().enumerate() {
        if w.size.is_some_and(|size| size > MAX_FILE_BYTES) {
            return Err(too_large());
        }
        let sha256 = match w.sha256 {
            Some(digest)
                if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) =>
            {
                return Err(ApiError::bad_request(
                    "sha256 must be a 64-character hex string",
                ))
            }
            digest => digest.map(|digest| digest.to_ascii_lowercase()),
        };
        let bytes = B64
            .decode(w.content.as_deref().unwrap_or_default())
            .map_err(|e| ApiError::bad_request(format!("invalid base64 content: {e}")))?;
        let start = w.start.unwrap_or(0);
        let end = w
            .end
            .unwrap_or_else(|| start.saturating_add(bytes.len() as u64));
        let size = w.size.unwrap_or(end);
        let valid_range = if size == 0 {
            start == 0 && end == 0
        } else {
            start < end && end <= size
        };
        if !valid_range {
            return Err(ApiError::bad_request(format!(
                "invalid chunk range [{start}, {end}) for size {size}"
            )));
        }
        let expected = end - start;
        if bytes.len() as u64 != expected {
            return Err(ApiError::bad_request_field(
                &format!("writes[{index}].end"),
                format!(
                    "writes[{index}] ({}): content is {} bytes but end - start is {expected} \
                     (start={start}, end={end}, size={size}). For a whole file of N bytes use \
                     start=0, end=N, size=N.",
                    w.path,
                    bytes.len()
                ),
            ));
        }
        total = total.saturating_add(bytes.len());
        if bytes.len() > MAX_CHUNK_BYTES || total > MAX_REQUEST_BYTES {
            return Err(too_large());
        }
        parsed.push(Chunk {
            path: w.path,
            size,
            upload_id: w.upload_id.unwrap_or_default(),
            bytes,
            start,
            end,
            sha256,
            secret: w.is_secret == Some(true),
        });
    }
    Ok(parsed)
}

/// A per-file failure, reported in that file's result.
struct FileError {
    code: &'static str,
    message: String,
    details: Option<serde_json::Value>,
}

impl FileError {
    fn bad_request(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            code: "bad_request",
            details: Some(serde_json::json!({
                "violations": [{ "field": "request", "message": message }]
            })),
            message,
        }
    }

    fn conflict() -> Self {
        Self {
            code: "conflict",
            message: "The request conflicts with the file's current state.".into(),
            details: Some(serde_json::json!({ "resource": "file" })),
        }
    }

    fn disk_full(message: impl Into<String>) -> Self {
        Self {
            code: "insufficient_resources",
            message: message.into(),
            details: Some(serde_json::json!({ "resource": "disk" })),
        }
    }

    fn internal(e: impl std::fmt::Display) -> Self {
        Self {
            code: "internal",
            message: e.to_string(),
            details: None,
        }
    }

    /// Why the VM refused the finished file, from what its write printed.
    fn from_guest(path: &str, error: &anyhow::Error) -> Self {
        let text = format!("{error:#}");
        if text.contains(crate::backend::IS_A_DIRECTORY) {
            Self::bad_request(format!("finalize {path}: Is a directory (os error 21)"))
        } else if text.contains("No space left on device") {
            Self::disk_full(format!(
                "write {path}: No space left on device (os error 28)"
            ))
        } else {
            Self::internal(text)
        }
    }
}

async fn write(s: &AppState, vm_id: &str, handle: &VmHandle, chunk: Chunk) -> SyncWriteResult {
    let mut result = SyncWriteResult {
        path: chunk.path.clone(),
        size: chunk.size,
        received_bytes: chunk.end - chunk.start,
        ranges: vec![ByteRange {
            start: chunk.start,
            end: chunk.end,
        }],
        complete: false,
        written: false,
        error: None,
    };
    match write_chunk(s, vm_id, handle, chunk).await {
        Ok(complete) => {
            result.complete = complete;
            result.written = complete;
        }
        Err(e) => {
            result.received_bytes = 0;
            result.ranges.clear();
            result.error = Some(SyncFileError {
                code: e.code,
                message: e.message,
                details: e.details,
                recovery: serde_json::json!({ "work": "unknown", "context": { "vm_id": vm_id } }),
            });
        }
    }
    result
}

/// Stages one chunk and, once its upload covers the whole file, writes the
/// file into the VM and forgets the upload: its id then starts a new one.
/// Returns whether the file was written.
async fn write_chunk(
    s: &AppState,
    vm_id: &str,
    handle: &VmHandle,
    chunk: Chunk,
) -> Result<bool, FileError> {
    // Without an id, or with nothing to send, the entry is a whole file of
    // its own.
    let one_shot = chunk.upload_id.is_empty() || chunk.start == chunk.end;
    let upload_id = if one_shot {
        ulid::Ulid::new().to_string()
    } else {
        chunk.upload_id.clone()
    };
    let key = (vm_id.to_owned(), upload_id);
    let staging = staging_path(s, &key);
    let range = (chunk.start, chunk.end);
    let digest: [u8; 32] = Sha256::digest(&chunk.bytes).into();
    let fresh = match admit_chunk(s, &key, &staging, &chunk, digest)? {
        Admission::Duplicate { complete } => return Ok(complete),
        Admission::Accepted { fresh } => fresh,
    };

    let staged = stage(&staging, fresh, chunk.start, &chunk.bytes).await;
    let complete = {
        let mut uploads = s.uploads.lock().expect("uploads lock");
        if let Err(e) = staged {
            release(&mut uploads, &key, range);
            return Err(FileError::internal(format!("stage {}: {e}", chunk.path)));
        }
        let upload = uploads
            .get_mut(&key)
            .ok_or_else(|| FileError::internal("upload expired"))?;
        upload.last_activity = Instant::now();
        let ranges = upload.chunks.iter().map(|(range, _)| *range).collect();
        drop(uploads);
        covers(ranges, chunk.size)
    };
    if !complete {
        return Ok(false);
    }

    let finished = finish(s, handle, &chunk, &staging).await;
    tokio::fs::remove_file(&staging).await.ok();
    s.uploads.lock().expect("uploads lock").remove(&key);
    finished.map(|()| true)
}

enum Admission {
    /// A new range of the upload; `fresh` when it starts the upload.
    Accepted { fresh: bool },
    /// The same bytes of a range already received.
    Duplicate { complete: bool },
}

/// Drops a range whose bytes never reached staging, and its upload when
/// that leaves it empty, so a retry is admitted.
fn release(
    uploads: &mut HashMap<(String, String), Upload>,
    key: &(String, String),
    range: (u64, u64),
) {
    if let Some(upload) = uploads.get_mut(key) {
        upload.chunks.retain(|(received, _)| *received != range);
        if upload.chunks.is_empty() {
            uploads.remove(key);
        }
    }
}

/// Starts the chunk's upload, or checks the chunk against the one it
/// continues: an upload is one file of one size, a range is received once,
/// and the same bytes sent again for a range are accepted again.
fn admit_chunk(
    s: &AppState,
    key: &(String, String),
    staging: &std::path::Path,
    chunk: &Chunk,
    digest: [u8; 32],
) -> Result<Admission, FileError> {
    let (start, end) = (chunk.start, chunk.end);
    let mut uploads = s.uploads.lock().expect("uploads lock");
    sweep_stale(&mut uploads);
    if let Some(upload) = uploads.get_mut(key) {
        if upload.path != chunk.path || upload.size != chunk.size {
            return Err(FileError::conflict());
        }
        if let Some((_, received)) = upload
            .chunks
            .iter()
            .find(|(range, _)| *range == (start, end))
        {
            if *received != digest {
                return Err(FileError::conflict());
            }
            let ranges = upload.chunks.iter().map(|(range, _)| *range).collect();
            return Ok(Admission::Duplicate {
                complete: covers(ranges, chunk.size),
            });
        }
        if upload
            .chunks
            .iter()
            .any(|((s, e), _)| start < *e && *s < end)
        {
            return Err(FileError::conflict());
        }
        upload.chunks.push(((start, end), digest));
        return Ok(Admission::Accepted { fresh: false });
    }
    let staging_limit = s.config.resources.as_ref().map_or(u64::MAX, |resources| {
        resources.disk_mib.saturating_mul(1 << 20)
    });
    let reserved = uploads
        .values()
        .try_fold(0u64, |total, upload| total.checked_add(upload.size))
        .unwrap_or(u64::MAX);
    if chunk.size > staging_limit.saturating_sub(reserved) {
        return Err(FileError::disk_full(format!(
            "sync uploads reserve {reserved} bytes of the {staging_limit}-byte disk budget"
        )));
    }
    uploads.insert(
        key.clone(),
        Upload {
            staging: staging.to_owned(),
            path: chunk.path.clone(),
            size: chunk.size,
            chunks: vec![((start, end), digest)],
            last_activity: Instant::now(),
        },
    );
    drop(uploads);
    Ok(Admission::Accepted { fresh: true })
}

fn staging_path(s: &AppState, (vm_id, upload_id): &(String, String)) -> PathBuf {
    let digest = Sha256::digest(format!("{vm_id}\0{upload_id}").as_bytes());
    s.config
        .data_dir
        .join("sync-staging")
        .join(hex::encode(digest))
}

async fn stage(
    staging: &std::path::Path,
    truncate: bool,
    start: u64,
    bytes: &[u8],
) -> std::io::Result<()> {
    if let Some(dir) = staging.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(truncate)
        .open(staging)
        .await?;
    file.seek(SeekFrom::Start(start)).await?;
    file.write_all(bytes).await?;
    file.flush().await
}

async fn finish(
    s: &AppState,
    handle: &VmHandle,
    chunk: &Chunk,
    staging: &std::path::Path,
) -> Result<(), FileError> {
    if let Some(expected) = &chunk.sha256 {
        let actual = file_sha256(staging).await.map_err(FileError::internal)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(FileError::bad_request(
                "upload checksum does not match the expected digest",
            ));
        }
    }
    let mode = if chunk.secret { 0o600 } else { 0o644 };
    s.backend
        .write_file(handle, &chunk.path, staging, mode)
        .await
        .map_err(|e| FileError::from_guest(&chunk.path, &e))
}

async fn file_sha256(path: &std::path::Path) -> std::io::Result<String> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            return Ok(hex::encode(hasher.finalize()));
        }
        hasher.update(&buf[..n]);
    }
}

fn sweep_stale(uploads: &mut HashMap<(String, String), Upload>) {
    uploads.retain(|_, upload| {
        let live = upload.last_activity.elapsed() <= UPLOAD_TTL;
        if !live {
            let _ = std::fs::remove_file(&upload.staging);
        }
        live
    });
}

/// Partial uploads cannot be resumed after a worker restart because their
/// received-range metadata is intentionally process-local.
pub fn clear_staging(data_dir: &std::path::Path) -> std::io::Result<()> {
    let dir = data_dir.join("sync-staging");
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// True when `ranges` together cover `[0, size)`.
fn covers(mut ranges: Vec<(u64, u64)>, size: u64) -> bool {
    ranges.sort_unstable();
    let mut covered = 0;
    for (start, end) in ranges {
        if start > covered {
            return false;
        }
        covered = covered.max(end);
    }
    covered >= size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_cover_out_of_order_and_not_across_a_gap() {
        assert!(!covers(vec![(10, 20)], 20));
        assert!(covers(vec![(10, 20), (0, 10)], 20));
        assert!(!covers(vec![(0, 5), (6, 20)], 20));
        assert!(covers(vec![], 0));
    }
}
