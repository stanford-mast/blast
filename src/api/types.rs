// API fields are part of the JSON contract and deserialized by serde;
// some are not yet consumed in Rust code, suppress dead_code for this module.
#![allow(dead_code)]

/// Control-plane-compatible request/response types, a minimal subset.
///
/// Field names and JSON representations match the control plane API so that
/// standard SDKs work against a BLAST endpoint without modification.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Fork ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ForkRequest {
    pub source_vm_id: Option<String>,
    pub source_vm_name: Option<String>,
    /// OCI image reference, e.g. "ubuntu:24.04".
    pub image: Option<String>,
    pub name: Option<String>,
    pub resources: Option<ResourcesInput>,
    pub network: Option<NetworkInput>,
    /// Credentials for pulling `image` from a private registry (ghcr.io, a
    /// private Docker Hub repo, ECR, etc). Used only for this one pull, never
    /// stored. Ignored when forking from a source VM, which has no registry
    /// to authenticate against.
    pub registry_auth: Option<RegistryAuth>,
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

#[derive(Debug, Deserialize)]
pub struct ResourcesInput {
    pub vcpu: Option<u32>,
    pub memory_mib: Option<u64>,
    pub disk_mib: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct NetworkInput {
    pub ssh_public_keys: Option<Vec<String>>,
}

// ── VM object ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct VmObject {
    pub vm_id: String,
    pub name: Option<String>,
    pub state: String,
    pub provider: String,
    pub region: String,
    pub platform: String,
    pub resources: ResourcesOutput,
}

#[derive(Debug, Serialize)]
pub struct ResourcesOutput {
    pub vcpu: u32,
    pub memory_mib: u64,
    pub disk_mib: u64,
}

// ── Run ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RunRequest {
    pub command: Option<String>,
    pub session_id: Option<String>,
    pub session_idx: Option<u32>,
    pub timeout: Option<u64>,
    /// Sync window in seconds. `0` returns a pollable command run immediately
    /// after successful dispatch without polling for completion. A positive
    /// value bounds the synchronous wait. Omission uses the 300-second
    /// default. This does not bound command execution; use `timeout` for the
    /// execution and kill bound. Values above approximately 350 seconds can
    /// exceed the load balancer idle limit.
    pub time_to_background: Option<u64>,
    pub env: Option<HashMap<String, String>>,
    pub cwd: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RunResponse {
    pub run_id: String,
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout_encoding: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr_encoding: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// System failure explanation when `state == "failed"`, distinct from
    /// `stderr`, which is the program's own error output. Absent for runs
    /// that are still running or completed normally.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fail_reason: Option<String>,
}

// ── Sessions ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    pub env: Option<HashMap<String, String>>,
    pub cwd: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct SessionObject {
    pub session_id: String,
    pub session_idx: u32,
    pub state: String,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
pub struct ListSessionsResponse {
    pub sessions: Vec<SessionObject>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub deleted: bool,
}

// ── Sync ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum SyncRequest {
    Read { path: String },
    Write { writes: Vec<SyncWrite> },
}

#[derive(Debug, Deserialize)]
pub struct SyncWrite {
    pub path: String,
    #[serde(default)]
    pub size: u64,
    pub upload_id: Option<String>,
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
        content: Option<String>,
        encoding: Option<String>,
        presigned_url: Option<String>,
        expires_in: Option<u64>,
        method: Option<String>,
    },
    Write {
        ok: bool,
        results: Vec<SyncWriteResult>,
    },
}

#[derive(Debug, Serialize)]
pub struct SyncWriteResult {
    pub received_bytes: u64,
    pub complete: bool,
    pub written: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presigned_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upload_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_in: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ── Regions ───────────────────────────────────────────────────────────────────

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

#[derive(Debug, Serialize)]
pub struct ListVmsResponse {
    pub vms: Vec<VmObject>,
}

// ── Worker registration (BLAST → control plane) ───────────────────────────────

#[derive(Debug, Serialize)]
pub struct WorkerRegisterRequest {
    pub worker_provider: String,
    pub worker_region: String,
    pub platform: String,
    /// "docker" / "smolvm" / "hypeman" -- the snapshot format this worker's
    /// VMs use, distinct from `platform` (CPU architecture, identical across
    /// backends on the same host). Informational for now; the control plane
    /// always routes an existing VM back to its original owning worker, so
    /// nothing currently picks a worker by backend -- but a future
    /// migrate-across-BYOC-workers feature would need it to avoid resuming a
    /// snapshot on a backend that can't read its format.
    pub backend: String,
    #[serde(rename = "cpu_count")]
    pub vcpu: u32,
    pub memory_mib: u64,
    pub disk_mib: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct WorkerRegisterResponse {
    pub worker_id: String,
}

#[derive(Debug, Serialize)]
pub struct WorkerHeartbeatRequest {
    pub vcpu: u32,
    pub memory_mib: u64,
    pub disk_mib: u64,
    pub vcpu_free: u32,
    pub memory_mib_free: u64,
    pub disk_mib_free: u64,
    pub vm_count: u32,
}

#[derive(Debug, Deserialize)]
pub struct WorkerCommand {
    pub command_id: String,
    pub command: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct WorkerCommandResult {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl WorkerCommandResult {
    pub fn err(msg: impl Into<String>) -> Self {
        Self { ok: false, result: None, error: Some(msg.into()) }
    }
}
