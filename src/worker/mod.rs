//! Remote control-plane membership and transport.

pub mod client;
pub mod heartbeat;
pub mod leave;
pub mod records;
pub mod tunnel;

use std::{
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::atomic::Ordering,
    time::Duration,
};

use anyhow::{Context, Result};
use axum::Router;
use tokio::{sync::watch, task::JoinHandle};
use tracing::{info, warn};

use crate::{api::handlers::AppState, config::RemoteConfig, snapshot::SnapshotStore};

use client::CallError;
use heartbeat::{Ended, WorkerState};

/// How long stopping waits for queued VM and run records to reach the
/// control plane.
const RECORDS_FLUSH_TIMEOUT: Duration = Duration::from_secs(10);

pub struct Worker {
    id: String,
    control_plane: client::ControlPlane,
    app: AppState,
    advertised: crate::backend::Resources,
    drain_timeout: Duration,
    heartbeat_task: JoinHandle<Ended>,
    tunnel_task: JoinHandle<()>,
    _membership_lock: File,
}

fn worker_id_path(data_dir: &Path) -> PathBuf {
    data_dir.join("worker_id")
}

fn incarnation_path(data_dir: &Path) -> PathBuf {
    data_dir.join("worker_incarnation")
}

pub fn host_identity() -> Result<String> {
    #[cfg(target_os = "linux")]
    if let Ok(id) = std::fs::read_to_string("/etc/machine-id") {
        let id = id.trim();
        if !id.is_empty() {
            return Ok(format!("linux:{id}"));
        }
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
            .context("read macOS platform UUID")?;
        let text = String::from_utf8_lossy(&output.stdout);
        if let Some(line) = text.lines().find(|line| line.contains("IOPlatformUUID")) {
            if let Some(id) = line.split('"').nth(3).filter(|id| !id.is_empty()) {
                return Ok(format!("macos:{id}"));
            }
        }
    }
    #[cfg(windows)]
    {
        let output = std::process::Command::new("reg")
            .args([
                "query",
                r"HKLM\SOFTWARE\Microsoft\Cryptography",
                "/v",
                "MachineGuid",
            ])
            .output()
            .context("read Windows machine GUID")?;
        let text = String::from_utf8_lossy(&output.stdout);
        if let Some(id) = text.split_whitespace().last().filter(|id| !id.is_empty()) {
            return Ok(format!("windows:{id}"));
        }
    }
    anyhow::bail!("cannot determine this host's stable machine identity")
}

pub fn bind_membership_to_host(data_dir: &Path) -> Result<()> {
    let path = data_dir.join("worker_host");
    let current = host_identity()?;
    match std::fs::read_to_string(&path) {
        Ok(stored) if stored.trim() == current => Ok(()),
        Ok(_) => anyhow::bail!(
            "{} belongs to another host; do not clone or share a connected worker data directory",
            data_dir.display()
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::write(&path, current).with_context(|| format!("write {}", path.display()))
        }
        Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
    }
}

pub fn lock_membership(data_dir: &Path) -> Result<File> {
    let path = data_dir.join("worker.lock");
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("open {}", path.display()))?;
    fs2::FileExt::try_lock_exclusive(&file).with_context(|| {
        format!(
            "lock {}; another BLAST process is already using this data directory",
            path.display()
        )
    })?;
    Ok(file)
}

pub fn lock_connected_membership(data_dir: &Path) -> Result<File> {
    let membership = lock_membership(data_dir)?;
    bind_membership_to_host(data_dir)?;
    Ok(membership)
}

pub fn new_incarnation() -> String {
    ulid::Ulid::new().to_string()
}

pub fn publish_incarnation(data_dir: &Path, incarnation: &str) -> Result<()> {
    let path = incarnation_path(data_dir);
    if std::fs::read_to_string(&path).is_ok_and(|current| current == incarnation) {
        return Ok(());
    }
    crate::durable::write_private_atomic(&path, incarnation.as_bytes())
        .with_context(|| format!("publish {}", path.display()))
}

pub fn current_incarnation(data_dir: &Path) -> Result<String> {
    let path = incarnation_path(data_dir);
    let incarnation = std::fs::read_to_string(&path)
        .with_context(|| format!("read admitted worker incarnation from {}", path.display()))?;
    let incarnation = incarnation.trim();
    ulid::Ulid::from_string(incarnation)
        .with_context(|| format!("{} does not hold a ULID", path.display()))?;
    Ok(incarnation.to_owned())
}

/// Reads the worker id from `{data_dir}/worker_id`, creating it on first
/// start. A file that holds anything but a ULID is an error, not a reason to
/// silently become a different worker.
pub fn load_or_create_worker_id(data_dir: &Path) -> Result<String> {
    let path = worker_id_path(data_dir);
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let id = text.trim();
            ulid::Ulid::from_string(id)
                .with_context(|| format!("{} does not hold a ULID", path.display()))?;
            Ok(id.to_owned())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let id = ulid::Ulid::new().to_string();
            std::fs::write(&path, &id).with_context(|| format!("write {}", path.display()))?;
            Ok(id)
        }
        Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
    }
}

/// Forgets this machine's membership in the pool, once it has left: its
/// local snapshots, which are stale for any VM another worker resumes, and
/// its worker id.
async fn forget_membership(snapshots: &SnapshotStore, data_dir: &Path) {
    if let Err(e) = snapshots.discard_worker_state().await {
        warn!(err = %format!("{e:#}"), "removing local snapshots failed");
    }
    let path = worker_id_path(data_dir);
    match std::fs::remove_file(&path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            warn!(err = %e, path = %path.display(), "removing the worker id failed");
        }
        _ => {}
    }
    let path = incarnation_path(data_dir);
    match std::fs::remove_file(&path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            warn!(err = %e, path = %path.display(), "removing the worker incarnation failed");
        }
        _ => {}
    }
}

impl Worker {
    /// Starts heartbeating, the tunnel, and sending `app.records`.
    /// `tunnel_router` serves requests arriving over the tunnel.
    pub fn start(
        cfg: &RemoteConfig,
        app: AppState,
        tunnel_router: Router,
        membership_lock: File,
        control_plane: client::ControlPlane,
    ) -> Result<Self> {
        bind_membership_to_host(&app.config.data_dir)?;
        let worker_id = load_or_create_worker_id(&app.config.data_dir)?;
        let advertised =
            heartbeat::advertised_resources(&app.total_resources, &app.config.data_dir);
        info!(%worker_id, pool_id = %cfg.pool_id, endpoint = %cfg.endpoint, "joining pool");

        let (url_tx, url_rx) = watch::channel(None);
        let heartbeat_task = tokio::spawn(heartbeat::run(heartbeat::Loop {
            control_plane: control_plane.clone(),
            app: app.clone(),
            advertised: advertised.clone(),
            tunnel_url_tx: url_tx,
        }));
        let tunnel_task = tokio::spawn(tunnel::run(url_rx, cfg.api_key.clone(), tunnel_router));

        Ok(Self {
            id: worker_id,
            control_plane,
            app,
            advertised,
            drain_timeout: Duration::from_secs(cfg.drain_timeout_secs),
            heartbeat_task,
            tunnel_task,
            _membership_lock: membership_lock,
        })
    }

    /// Resolves when heartbeating stops: the pool is being deleted, or the
    /// control plane refused this worker.
    pub async fn ended(&mut self) -> Ended {
        (&mut self.heartbeat_task)
            .await
            .expect("the heartbeat loop panicked")
    }

    /// Why the control plane refused this worker, for the operator.
    pub fn refusal(&self, status: reqwest::StatusCode, body: &str) -> anyhow::Error {
        anyhow::anyhow!(
            "the control plane refused worker {} (platform {}) in pool {}: {status}: {body}",
            self.id,
            self.app.platform,
            self.app.pool_id.as_deref().unwrap_or_default(),
        )
    }

    /// Drains and stops while retaining membership identity.
    pub async fn stop(self) {
        self.app.draining.store(true, Ordering::SeqCst);
        let _barrier = self.app.lifecycle_gate.write().await;
        let _ = self.report(WorkerState::Draining, 1).await;
        if let Err(e) = crate::lifecycle::hand_back(&self.app, self.drain_timeout, false).await {
            warn!(err = %format!("{e:#}"), "stopped without completing VM hand-back; restart this worker to recover its VMs");
        }
        self.flush_records().await;
        self.heartbeat_task.abort();
        self.tunnel_task.abort();
        info!(worker_id = %self.id, "stopped; a restart rejoins the pool as this worker");
    }

    /// Forgets a membership the control plane retired. The VMs it held are
    /// other workers' now, so their machines and local state go with it.
    pub async fn retire(self) {
        self.heartbeat_task.abort();
        self.tunnel_task.abort();
        let _barrier = self.app.lifecycle_gate.write().await;
        for vm in self.app.store.all().await {
            if let Some(handle) = &vm.handle {
                if let Err(e) = self.app.backend.delete(handle).await {
                    warn!(vm_id = %vm.vm_id, err = %format!("{e:#}"), "removing a retired worker's machine failed");
                }
            }
            self.app.store.remove(&vm.vm_id).await;
            if let Err(e) =
                crate::store::forget_portable_vm_state(&vm.vm_id, &self.app.store, &self.app.runs)
                    .await
            {
                warn!(vm_id = %vm.vm_id, err = %format!("{e:#}"), "forgetting a retired worker's VM failed");
            }
        }
        forget_membership(&self.app.snapshots, &self.app.config.data_dir).await;
        info!(worker_id = %self.id, "forgot the retired worker id");
    }

    /// Drains and permanently leaves the pool.
    pub async fn leave(self, require_transferable: bool) -> bool {
        self.app.draining.store(true, Ordering::SeqCst);
        let _barrier = self.app.lifecycle_gate.write().await;
        let _ = self.report(WorkerState::Draining, 1).await;
        if let Err(e) =
            crate::lifecycle::hand_back(&self.app, self.drain_timeout, require_transferable).await
        {
            warn!(err = %format!("{e:#}"), "could not safely hand back every VM; worker remains draining");
            return false;
        }
        self.flush_records().await;
        let reported = self.report(WorkerState::Left, 3).await;
        self.heartbeat_task.abort();
        self.tunnel_task.abort();
        match reported {
            Ok(()) => info!("left the pool"),
            Err(CallError::Rejected { status, .. }) if heartbeat::refuses_for_good(status) => {
                info!(%status, "the control plane no longer knows this worker; it has left");
            }
            Err(e) => {
                warn!(err = %e, "could not report leaving; run `blast leave` to finish");
                return false;
            }
        }
        forget_membership(&self.app.snapshots, &self.app.config.data_dir).await;
        true
    }

    async fn flush_records(&self) {
        if !self.app.records.flush(RECORDS_FLUSH_TIMEOUT).await {
            warn!("stopped with records still unsent");
        }
    }

    async fn report(&self, state: WorkerState, attempts: u32) -> Result<(), CallError> {
        let mut attempt = 1;
        loop {
            let body = heartbeat::build(&self.app, &self.advertised, state).await;
            match self.control_plane.heartbeat(&body).await {
                Ok(_) => return Ok(()),
                Err(e @ CallError::Rejected { .. }) => return Err(e),
                Err(e) if attempt == attempts => return Err(e),
                Err(e) => {
                    warn!(attempt, ?state, err = %format!("{e:#}"), "heartbeat failed");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    attempt += 1;
                }
            }
        }
    }
}
