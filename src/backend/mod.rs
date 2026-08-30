pub mod bootstrap;
pub mod docker;
pub mod hypeman;
pub mod smolvm;

pub fn host_platform() -> &'static str {
    match std::env::consts::ARCH {
        "aarch64" => "arm64",
        _ => "x86_64",
    }
}

use std::{collections::HashMap, path::Path, time::Duration};

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub use crate::api::types::RegistryAuth;

#[derive(Debug, Clone)]
pub struct VmHandle {
    pub id: String,
    pub platform: String,
}

/// Extract the registry host from an OCI image reference, following the same
/// convention Docker and other OCI tooling use: the first path segment before
/// a `/` is a registry host only when it looks like one (contains `.` or `:`,
/// or is exactly `localhost`); otherwise the reference is an (optionally
/// namespaced) Docker Hub image, and `"docker.io"` is returned.
///
/// Examples: `"ghcr.io/org/image"` -> `"ghcr.io"`; `"ubuntu:24.04"` ->
/// `"docker.io"`; `"myuser/private-repo"` -> `"docker.io"`;
/// `"registry.example.com:5000/image"` -> `"registry.example.com:5000"`.
pub fn extract_registry(image: &str) -> &str {
    if let Some(slash) = image.find('/') {
        let candidate = &image[..slash];
        if candidate.contains('.') || candidate.contains(':') || candidate == "localhost" {
            return candidate;
        }
    }
    "docker.io"
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resources {
    pub vcpu: u32,
    pub memory_mib: u64,
    pub disk_mib: u64,
}

#[derive(Debug)]
pub struct RunOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: i32,
}

#[async_trait]
pub trait VmBackend: Send + Sync + 'static {
    /// Boot a fresh VM from an OCI image reference. `registry_auth`, when
    /// present, authenticates the pull against a private registry.
    async fn fork_image(
        &self,
        image: &str,
        resources: &Resources,
        registry_auth: Option<&RegistryAuth>,
    ) -> Result<VmHandle>;

    /// Restore a VM from a snapshot directory.
    async fn fork_snapshot(&self, snap_dir: &Path, resources: &Resources) -> Result<VmHandle>;

    /// Run a command inside a running VM within a named session context.
    async fn run(
        &self,
        vm: &VmHandle,
        command: &str,
        session_id: &str,
        env: &HashMap<String, String>,
        cwd: &str,
        timeout: Duration,
    ) -> Result<RunOutput>;

    /// Freeze the VM: stop CPU execution while keeping memory resident.
    /// Backends that do not support a distinct pause may fall through to snapshot+delete.
    async fn pause(&self, vm: &VmHandle) -> Result<()>;

    /// Resume a paused VM (CPU unfrozen; memory already resident).
    async fn unpause(&self, vm: &VmHandle) -> Result<()>;

    /// Write VM state to `snap_dir`, leaving the VM running (dirty-sync snapshot).
    async fn snapshot(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()>;

    /// Suspend: write state to `snap_dir` and destroy the VM process.
    async fn suspend(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()>;

    /// Restore a suspended VM from `snap_dir`.
    async fn resume(&self, snap_dir: &Path, resources: &Resources) -> Result<VmHandle>;

    /// Destroy the VM and release all resources.
    async fn delete(&self, vm: &VmHandle) -> Result<()>;

    /// e.g. "linux/aarch64".
    fn platform(&self) -> &str;

    /// e.g. "docker", "smolvm", "hypeman". Unlike `platform` (CPU
    /// architecture, identical across backends on the same host) this
    /// identifies the snapshot FORMAT: a Docker commit-based image, a
    /// SmolVM memory snapshot, and a Hypeman snapshot are mutually
    /// unreadable, so a snapshot written by one backend can never be
    /// resumed by another. See `SnapshotStore::{write,check}_backend_marker`.
    fn kind(&self) -> &'static str;
}
