//! Request and response bodies, in the control plane's wire shapes (its
//! `openapi.json`) for the subset BLAST serves, so the control plane and its
//! SDKs work against a BLAST endpoint unchanged. The control plane forwards the
//! customer's fork and run bodies as it received them, placement fields
//! included, so those two ignore fields they don't know; fields the API
//! defines but BLAST cannot honour are parsed and refused by the handler
//! with `unsupported_operation`.

// Accepted-but-unused request fields are part of the parsed shape without
// ever being read, and field names are the wire names.
#![allow(dead_code, clippy::struct_field_names)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Deserialize)]
pub struct ForkRequest {
    pub source_vm_id: Option<String>,
    pub source_vm_name: Option<String>,
    /// OCI image reference, e.g. "ubuntu:24.04".
    pub image: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub resources: Option<ResourcesInput>,
    /// Credentials for pulling `image` from a private registry (ghcr.io, a
    /// private Docker Hub repo, ECR, etc). Used only for this one pull, never
    /// stored. Ignored when forking from a source VM, which has no registry
    /// to authenticate against.
    pub registry_auth: Option<RegistryAuth>,
    /// Seconds to wait for pool capacity; the worker's
    /// `admission_queue_secs` when absent.
    pub queueing_timeout: Option<u64>,
    pub platforms: Option<Vec<String>>,
    pub layers: Option<Vec<String>>,

    // Fields this worker refuses unless they ask for the default.
    pub dockerfile: Option<String>,
    pub source_org_id: Option<String>,
    pub source_org_name: Option<String>,
    pub nestedvirt: Option<bool>,
    pub public: Option<bool>,
    pub ssh_public_keys: Option<serde_json::Value>,
    pub disk: Option<bool>,
    pub durable: Option<bool>,
    pub policies: Option<serde_json::Value>,

    /// The request as it arrived, without its registry credentials: what the
    /// fork's activity is recorded with.
    #[serde(skip)]
    pub recorded: Option<String>,
}

/// Credentials for pulling `image` from a private registry, matching Docker's
/// own registry-auth shape.
///
/// For AWS ECR use username `AWS` with the output of `aws ecr
/// get-login-password`; for Google Artifact Registry use `oauth2accesstoken`
/// with an access token; for Docker Hub or GHCR use your username and a
/// personal access token. Short-lived tokens are fine -- the pull happens
/// once, at fork time.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryAuth {
    /// Registry username, e.g. `AWS` for ECR.
    pub username: String,
    /// Registry password or token. Never logged: `Debug` is implemented by
    /// hand below to redact it.
    pub password: String,
}

impl std::fmt::Debug for RegistryAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryAuth")
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .finish()
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourcesInput {
    pub vcpu: Option<u32>,
    pub memory_mib: Option<u64>,
    pub disk_mib: Option<u64>,
    pub vgpu: Option<serde_json::Value>,
}

/// The `Vm` schema.
#[derive(Debug, Serialize)]
pub struct Vm {
    pub vm_id: String,
    pub owner_org_id: String,
    pub created_at: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub public: bool,
    pub state: &'static str,
    pub region: Option<String>,
    pub provider: Option<String>,
    pub platform: String,
    pub sessions: Vec<Session>,
    pub resources: VmResources,
    pub network: VmNetwork,
    pub pool_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VmResources {
    pub vcpu: u32,
    pub memory_mib: u64,
    pub disk_mib: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct VmNetwork {
    pub ssh_public_keys: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ListVmsResponse {
    pub vms: Vec<Vm>,
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub deleted: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct RunRequest {
    pub command: Option<String>,
    pub session_id: Option<String>,
    pub session_idx: Option<u32>,
    /// Seconds before the command is killed; absent or `0` means no limit.
    pub timeout: Option<u64>,
    /// Sync window in seconds. `0` returns a pollable run immediately; a
    /// positive value bounds the synchronous wait (default 300). This does
    /// not bound command execution; `timeout` does.
    pub time_to_background: Option<u64>,
    /// Deprecated alias for `time_to_background: 0`.
    pub background: Option<bool>,
    pub queueing_timeout: Option<u64>,
    pub end_symbol: Option<String>,
    pub end_after_all_subprocesses: Option<bool>,
    pub signal: Option<String>,
    pub policies: Option<serde_json::Value>,
    /// BLAST extensions: per-run environment and working directory.
    pub env: Option<HashMap<String, String>>,
    pub cwd: Option<String>,
}

impl RunRequest {
    /// Whether the request says anything about a run beyond its command; one
    /// that says nothing at all only wakes the VM.
    pub const fn names_a_run(&self) -> bool {
        self.session_id.is_some()
            || self.session_idx.is_some()
            || self.timeout.is_some()
            || self.time_to_background.is_some()
            || self.background.is_some()
            || self.queueing_timeout.is_some()
            || self.end_symbol.is_some()
            || self.end_after_all_subprocesses.is_some()
            || self.env.is_some()
            || self.cwd.is_some()
    }
}

/// `createRun`'s body: complete output when the run finished inside the
/// sync window (HTTP 200), else just the run to poll (HTTP 202).
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum RunResponse {
    Completed {
        #[serde(skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        run_id: Option<String>,
        state: &'static str,
        stdout: String,
        stdout_encoding: &'static str,
        /// Present when the command printed more than `stdout` holds.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        stdout_truncated: bool,
        stderr: String,
        stderr_encoding: &'static str,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        stderr_truncated: bool,
        exit_code: Option<i32>,
    },
    Background {
        session_id: Option<String>,
        run_id: String,
        state: &'static str,
    },
}

impl RunResponse {
    /// The answer to a request that ran no command: no run and no session,
    /// and 0 is the acknowledgement's own status.
    pub const fn acknowledged() -> Self {
        Self::Completed {
            session_id: None,
            run_id: None,
            state: "completed",
            stdout: String::new(),
            stdout_encoding: "utf-8",
            stdout_truncated: false,
            stderr: String::new(),
            stderr_encoding: "utf-8",
            stderr_truncated: false,
            exit_code: Some(0),
        }
    }
}

/// The `Run` schema (`getRun`).
#[derive(Debug, Serialize)]
pub struct Run {
    pub run_id: String,
    pub vm_id: String,
    pub session_id: Option<String>,
    pub command: String,
    pub state: &'static str,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub exit_code: Option<i32>,
    pub fail_reason: Option<String>,
    pub stdout: String,
    pub stdout_encoding: &'static str,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stdout_truncated: bool,
    pub stderr: String,
    pub stderr_encoding: &'static str,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stderr_truncated: bool,
}

#[derive(Debug, Serialize)]
pub struct ListRunsResponse {
    pub runs: Vec<Run>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CancelRunResponse {
    pub cancelled: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateSessionRequest {
    pub env: Option<HashMap<String, String>>,
    pub cwd: Option<String>,
    // Terminal sessions, which this worker refuses.
    pub pty: Option<bool>,
    pub cols: Option<u32>,
    pub rows: Option<u32>,
    pub command: Option<String>,
}

/// The `Session` schema.
#[derive(Debug, Serialize, Clone)]
pub struct Session {
    pub session_id: String,
    pub session_idx: u32,
    pub state: &'static str,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
pub struct ListSessionsResponse {
    pub sessions: Vec<Session>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase", deny_unknown_fields)]
pub enum SyncRequest {
    Read { path: String },
    Write { writes: Vec<SyncWrite> },
    Manifest { path: String },
}

/// One entry of a write: `SyncChunkWrite` (a byte range
/// `[start, end)` of a `size`-byte file, sharing `upload_id` across chunks),
/// or a whole file when `upload_id`/`start`/`end` are omitted. The API's
/// presigned forms are recognised and refused.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyncWrite {
    pub path: String,
    pub size: Option<u64>,
    pub upload_id: Option<String>,
    /// Base64-encoded bytes.
    pub content: Option<String>,
    pub start: Option<u64>,
    pub end: Option<u64>,
    pub sha256: Option<String>,
    pub is_secret: Option<bool>,
    pub presigned: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum SyncResponse {
    Read {
        ok: bool,
        path: String,
        size: u64,
        content: String,
        /// `utf8` or `base64`.
        encoding: &'static str,
    },
    Write {
        ok: bool,
        results: Vec<SyncWriteResult>,
    },
}

/// `SyncChunkWriteResult`.
#[derive(Debug, Serialize)]
pub struct SyncWriteResult {
    pub path: String,
    pub size: u64,
    pub received_bytes: u64,
    pub ranges: Vec<ByteRange>,
    pub complete: bool,
    pub written: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<SyncFileError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

/// The per-file `SyncFileError`.
#[derive(Debug, Serialize)]
pub struct SyncFileError {
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    /// What became of the write: unknown, for the VM named.
    pub recovery: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct RegionEntry {
    pub provider: String,
    pub region: String,
    pub endpoint: String,
    pub platform: String,
    pub vcpu: u32,
    pub memory_mib: u64,
    pub disk_mib: u64,
}

#[derive(Debug, Serialize)]
pub struct ListRegionsResponse {
    pub regions: Vec<RegionEntry>,
}
