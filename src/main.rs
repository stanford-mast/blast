#![deny(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::wildcard_imports)]

mod admission;
mod api;
mod backend;
mod config;
mod durable;
mod lifecycle;
mod snapshot;
mod store;
mod worker;

use std::sync::{atomic::AtomicBool, Arc};

use anyhow::{Context, Result};
use axum::Router;
use clap::{Parser, Subcommand};
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
};
use tracing::info;
use tracing_subscriber::EnvFilter;

use api::handlers::AppState;
use backend::{
    docker::DockerBackend, hypeman::HypemanBackend, smolvm::SmolvmBackend, Resources, VmBackend,
};
use config::{BackendConfig, Config};
use snapshot::{SnapshotStore, Storage};
use store::{RunStore, Store};
use worker::{heartbeat::Ended, records::Records, Worker};

#[derive(Parser)]
#[command(name = "blast", about = "Sandbox serving engine")]
struct Cli {
    #[arg(long, env = "BLAST_CONFIG", global = true)]
    config: Option<std::path::PathBuf>,
    /// Additional execution-compatibility suffix; wins over `[worker] platform`.
    #[arg(long, global = true)]
    platform: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Leave the pool for good: the running BLAST hands its VMs back,
    /// reports `left`, forgets its worker id and exits (with none running,
    /// the worker id is reported `left` and forgotten). Stopping BLAST
    /// instead keeps it in the pool.
    Leave,
}

/// Why a connected worker exits.
enum Exit {
    /// A signal: stop, and rejoin as the same worker on restart.
    Stop,
    /// The operator ran `blast leave` (answered on the sender with whether
    /// the worker left), or the control plane is deleting the pool.
    Leave(Option<oneshot::Sender<bool>>),
    Refused {
        status: reqwest::StatusCode,
        body: String,
    },
    /// The control plane retired this worker id.
    Retired,
    MembershipLeaseExpired,
}

fn configured_platform(cli: &Cli, cfg: &Config) -> Result<String> {
    Ok(backend::worker_platform(
        cli.platform.as_deref().or(cfg.worker.platform.as_deref()),
        &cfg.backend,
        &worker::host_identity()?,
    ))
}

async fn backend_and_snapshots(
    cfg: &mut Config,
    remote: bool,
) -> Result<(Arc<dyn VmBackend>, Arc<SnapshotStore>)> {
    backend::bootstrap::ensure_backend_ready(&mut cfg.backend, &cfg.data_dir)?;
    let backend: Arc<dyn VmBackend> = match &cfg.backend {
        BackendConfig::Docker => Arc::new(DockerBackend::new()),
        BackendConfig::Hypeman {
            endpoint,
            token,
            cli_binary,
        } => Arc::new(HypemanBackend::new(
            endpoint.clone(),
            token.clone(),
            cli_binary.clone(),
        )),
        BackendConfig::Smolvm { binary } => {
            Arc::new(SmolvmBackend::new(binary.clone(), cfg.data_dir.clone()))
        }
    };
    let storage = match (&cfg.worker.storage_dir, remote) {
        (Some(dir), _) => {
            tokio::fs::create_dir_all(dir).await?;
            Storage::Dir(dir.clone())
        }
        (None, true) => Storage::object(cfg.worker.object_storage_endpoint.clone()),
        (None, false) => Storage::None,
    };
    let shared = match &storage {
        Storage::None => false,
        Storage::Dir(_) => cfg.worker.storage_dir_shared,
        Storage::Object { .. } => true,
    };
    Ok((
        backend,
        Arc::new(SnapshotStore::with_shared_storage(
            &cfg.data_dir,
            storage,
            shared,
        )),
    ))
}

fn configured_resources(cfg: &Config) -> Resources {
    cfg.resources.as_ref().map_or(
        Resources {
            vcpu: 0,
            memory_mib: 0,
            disk_mib: 0,
        },
        |resources| Resources {
            vcpu: resources.vcpu,
            memory_mib: resources.memory_mib,
            disk_mib: resources.disk_mib,
        },
    )
}

async fn restore_state(state: &AppState) -> Result<()> {
    state.snapshots.scavenge_transient_snapshots().await?;
    if state.allow_local_snapshot_adoption {
        state.snapshots.adopt_all_local_manifests().await?;
    }
    // A removal is recorded only once the control plane has accepted the
    // delete, so finishing one needs nothing from it: a worker that has not
    // joined yet could not ask.
    for vm_id in state.snapshots.pending_removals().await? {
        state.snapshots.forget(&vm_id).await?;
    }
    for manifest in state.snapshots.owned_manifests().await? {
        let restored = api::handlers::record_from_manifest(state, manifest)
            .await
            .map_err(|error| anyhow::anyhow!("restore owned VM: {error:?}"))?;
        if let Some(record) = restored {
            state.store.insert(record).await;
        }
    }
    for (run_id, run) in state.runs.all().await {
        state.records.run(&run_id, &run)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            // A worker's joins, heartbeat failures and tunnel drops are what an
            // operator looks for first, so info shows without RUST_LOG.
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    loop {
        let runtime = tokio::runtime::Runtime::new()?;
        let served = runtime.block_on(serve(&cli));
        // Ends every task of the worker that just stopped, so none of them
        // outlives the worker id it served.
        runtime.shutdown_timeout(RUNTIME_SHUTDOWN);
        match served? {
            Served::Stopped => return Ok(()),
            Served::Retired => info!("the worker id has left the pool; joining as a new worker"),
            Served::LeaseExpired => {
                info!("the membership lease expired; rejoining the pool as this worker");
            }
        }
    }
}

/// How long a stopped worker's tasks get to end.
const RUNTIME_SHUTDOWN: std::time::Duration = std::time::Duration::from_secs(10);

/// How one worker's service ended.
enum Served {
    Stopped,
    /// The control plane retired the worker id; its membership is forgotten
    /// and the next start joins as a new worker.
    Retired,
    /// The control plane went unreachable for the length of the membership
    /// lease, so the worker fenced itself. The next start rejoins under a new
    /// incarnation and keeps the VMs the control plane still counts as its own.
    LeaseExpired,
}

/// The app state over `cfg.data_dir`, with what it restored from there.
async fn build_state(
    cfg: &mut Config,
    remote: Option<&config::RemoteConfig>,
    platform: String,
    control_plane: worker::client::ControlPlane,
) -> Result<AppState> {
    let records = Records::new(&cfg.data_dir, control_plane.clone())?;
    api::sync::clear_staging(&cfg.data_dir)?;
    let (backend, snapshots) = backend_and_snapshots(cfg, remote.is_some()).await?;
    let state = AppState {
        store: Store::durable(&cfg.data_dir, backend.vm_memory_overhead_mib())?,
        runs: RunStore::durable(&cfg.data_dir)?,
        backend,
        snapshots,
        config: Arc::new(cfg.clone()),
        total_resources: configured_resources(cfg),
        records,
        control_plane,
        allow_local_snapshot_adoption: remote.is_none(),
        platform,
        pool_id: remote.map(|config| config.pool_id.clone()),
        draining: Arc::new(AtomicBool::new(false)),
        lifecycle_gate: Arc::new(tokio::sync::RwLock::new(())),
        vm_locks: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        uploads: Arc::default(),
        run_cancels: Arc::default(),
    };
    restore_state(&state).await?;
    Ok(state)
}

async fn serve(cli: &Cli) -> Result<Served> {
    let mut cfg = Config::load(cli.config.as_deref())?;
    let remote = cfg.worker.remote()?;
    cfg.validate_resources(remote.is_some())?;
    let platform = configured_platform(cli, &cfg)?;

    if matches!(cli.command, Some(Command::Leave)) {
        let remote = remote
            .context("`blast leave` needs [worker] control_plane_endpoint, api_key and pool_id")?;
        let advertised =
            worker::heartbeat::advertised_resources(&configured_resources(&cfg), &cfg.data_dir);
        worker::leave::run(&cfg, &remote, &platform, &advertised).await?;
        return Ok(Served::Stopped);
    }

    tokio::fs::create_dir_all(&cfg.data_dir).await?;
    let process_lock = if remote.is_some() {
        worker::lock_connected_membership(&cfg.data_dir)?
    } else {
        worker::lock_membership(&cfg.data_dir)?
    };
    let control_plane = if let Some(remote) = &remote {
        let worker_id = worker::load_or_create_worker_id(&cfg.data_dir)?;
        let incarnation = worker::new_incarnation();
        worker::client::ControlPlane::new(remote, &worker_id, &incarnation)
    } else {
        worker::client::ControlPlane::local(&cfg.data_dir)?
    };
    let state = build_state(&mut cfg, remote.as_ref(), platform, control_plane.clone()).await?;
    let records_task = tokio::spawn(state.records.clone().send_loop());

    lifecycle::spawn(&cfg.lifecycle, &state);

    let worker = if let Some(c) = &remote {
        let mut tunnel_state = state.clone();
        tunnel_state.allow_local_snapshot_adoption = true;
        Some(Worker::start(
            c,
            state.clone(),
            api::router(tunnel_state, None),
            process_lock,
            control_plane,
        )?)
    } else {
        info!("using embedded local control plane");
        None
    };

    let local_token = cfg.api.token.as_deref().map(Arc::from);
    if local_token.is_none() {
        if remote.is_some() {
            tracing::warn!("[api] token is not set: only liveness diagnostics are enabled");
        } else {
            tracing::warn!(
                "[api] token is not set: the local API on port {} is unauthenticated",
                cfg.port
            );
        }
    }
    let (leave_tx, leave_rx) = mpsc::channel(1);
    let router = if remote.is_some() {
        api::diagnostics_router(state, local_token, Router::new())
    } else {
        api::router_with(state, local_token, worker::leave::route(leave_tx))
    }
    .layer(tower_http::trace::TraceLayer::new_for_http());

    let host = if remote.is_some() {
        "127.0.0.1"
    } else {
        "0.0.0.0"
    };
    let addr = format!("{host}:{}", cfg.port);
    let listener = TcpListener::bind(&addr).await?;
    info!(addr, "blast listening");
    let (ended_tx, mut ended_rx) = oneshot::channel();
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown(worker, leave_rx, ended_tx))
        .await?;
    records_task.abort();

    ended_rx.try_recv().unwrap_or(Ok(Served::Stopped))
}

/// Resolves once BLAST should stop: on SIGTERM or Ctrl-C, on `blast leave`,
/// when the control plane deletes a connected worker's pool, or when it
/// refuses the worker or retires its id. A connected worker first stops (a
/// signal), leaves (`blast leave`, pool deleted) or forgets its retired
/// membership; a refusal or a retirement is sent on `ended`. The local API
/// keeps serving meanwhile.
async fn shutdown(
    mut worker: Option<Worker>,
    mut leave_requests: mpsc::Receiver<oneshot::Sender<bool>>,
    ended: oneshot::Sender<Result<Served>>,
) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    let worker_ended = async {
        match &mut worker {
            Some(w) => w.ended().await,
            None => std::future::pending().await,
        }
    };

    let exit = tokio::select! {
        () = ctrl_c => { info!("interrupted"); Exit::Stop }
        () = terminate => { info!("SIGTERM received"); Exit::Stop }
        ended = worker_ended => match ended {
            Ended::PoolDeleted => { info!("the pool is being deleted"); Exit::Leave(None) }
            Ended::Refused { status, body } => Exit::Refused { status, body },
            Ended::Retired => Exit::Retired,
            Ended::LeaseExpired => Exit::MembershipLeaseExpired,
        },
        Some(left) = leave_requests.recv() => { info!("asked to leave the pool"); Exit::Leave(Some(left)) }
    };

    let Some(worker) = worker else { return };
    match exit {
        Exit::Stop => {
            info!("stopping");
            worker.stop().await;
        }
        Exit::Leave(answer) => {
            info!("leaving the pool");
            // An operator-requested leave must preserve every VM for another
            // worker. Pool deletion is explicitly destructive and may retire
            // host-local/nonportable state.
            let left = worker.leave(answer.is_some()).await;
            if let Some(answer) = answer {
                let _ = answer.send(left);
            }
        }
        Exit::Refused { status, body } => {
            let _ = ended.send(Err(worker.refusal(status, &body)));
        }
        Exit::Retired => {
            worker.retire().await;
            let _ = ended.send(Ok(Served::Retired));
        }
        Exit::MembershipLeaseExpired => {
            let _ = ended.send(Ok(Served::LeaseExpired));
        }
    }
}
