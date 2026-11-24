//! `blast leave`: the operator takes this machine out of its pool for good.
//!
//! With BLAST running, the command asks that process to leave over the local
//! API (`POST /v1/worker/leave`) and waits until it has: the process hands
//! its VMs back, reports `left`, forgets its worker id, and exits. With none
//! running, the command reports `left` for the worker id in `{data_dir}`
//! itself, then forgets it and the local snapshots; the VMs' stored snapshots
//! stay with the pool.

use std::path::Path;

use anyhow::{bail, Context, Result};
use axum::{http::StatusCode, routing::post, Router};
use tokio::sync::{mpsc, oneshot};
use tracing::info;

use crate::{
    api::{error::ApiError, handlers::AppState},
    backend::Resources,
    config::{Config, RemoteConfig},
    snapshot::{SnapshotStore, Storage},
};

use super::{
    bind_membership_to_host,
    client::{CallError, ControlPlane},
    current_incarnation, forget_membership,
    heartbeat::{self, WorkerState},
    load_or_create_worker_id, lock_membership, worker_id_path,
};

pub const PATH: &str = "/v1/worker/leave";

/// Leave requests from the local API, each answered with whether the worker
/// left.
pub type Requests = mpsc::Sender<oneshot::Sender<bool>>;

/// The leave route, for the local listener only: the control plane never
/// makes a worker leave through the tunnel.
pub fn route(requests: Requests) -> Router<AppState> {
    Router::new().route(PATH, post(move || request(requests.clone())))
}

async fn request(requests: Requests) -> Result<StatusCode, ApiError> {
    let (left_tx, left_rx) = oneshot::channel();
    requests
        .send(left_tx)
        .await
        .map_err(|_| ApiError::unavailable("the worker is already stopping"))?;
    match left_rx.await {
        Ok(true) => Ok(StatusCode::NO_CONTENT),
        _ => Err(ApiError::unavailable(
            "the worker stopped without reporting `left`; run `blast leave` again",
        )),
    }
}

/// Runs `blast leave` for the BLAST `cfg` configures.
pub async fn run(
    cfg: &Config,
    remote: &RemoteConfig,
    platform: &str,
    advertised: &Resources,
) -> Result<()> {
    if ask_running_worker(cfg).await? {
        info!("the running BLAST left the pool");
        return Ok(());
    }
    leave_stopped_worker(remote, &cfg.data_dir, platform, advertised).await
}

/// Asks the BLAST listening on `cfg.port` to leave; false when none is.
async fn ask_running_worker(cfg: &Config) -> Result<bool> {
    let mut request = reqwest::Client::new().post(format!("http://127.0.0.1:{}{PATH}", cfg.port));
    if let Some(token) = &cfg.api.token {
        request = request.bearer_auth(token);
    }
    let resp = match request.send().await {
        Ok(resp) => resp,
        Err(e) if e.is_connect() => return Ok(false),
        Err(e) => return Err(e).context("asking the running BLAST to leave"),
    };
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("the running BLAST did not leave: {status}: {body}");
    }
    Ok(true)
}

async fn leave_stopped_worker(
    remote: &RemoteConfig,
    data_dir: &Path,
    platform: &str,
    advertised: &Resources,
) -> Result<()> {
    if !worker_id_path(data_dir).exists() {
        info!(data_dir = %data_dir.display(), "no worker id: this machine is not in the pool");
        return Ok(());
    }
    let _membership_lock = lock_membership(data_dir)?;
    bind_membership_to_host(data_dir)?;
    let worker_id = load_or_create_worker_id(data_dir)?;
    let in_use = Resources {
        vcpu: 0,
        memory_mib: 0,
        disk_mib: 0,
    };
    let incarnation = current_incarnation(data_dir)?;
    let control_plane = ControlPlane::new(remote, &worker_id, &incarnation);
    let claim = heartbeat::body(platform, advertised, &in_use, WorkerState::Draining);
    match control_plane.heartbeat(&claim).await {
        Ok(_) => {}
        Err(CallError::Rejected { status, body }) if heartbeat::refuses_for_good(status) => {
            info!(%worker_id, %status, %body, "the control plane no longer takes this worker; forgetting it");
            forget_membership(&SnapshotStore::new(data_dir, Storage::None), data_dir).await;
            return Ok(());
        }
        Err(error) => {
            return Err(error).context("could not claim the stopped worker before leaving");
        }
    }
    let body = heartbeat::body(platform, advertised, &in_use, WorkerState::Left);
    match control_plane.heartbeat(&body).await {
        Ok(_) => info!(%worker_id, "left the pool"),
        Err(CallError::Rejected { status, body }) if heartbeat::refuses_for_good(status) => {
            info!(%worker_id, %status, %body, "the control plane no longer takes this worker; forgetting it");
        }
        Err(e) => return Err(e).context("reporting `left`"),
    }
    forget_membership(&SnapshotStore::new(data_dir, Storage::None), data_dir).await;
    Ok(())
}
