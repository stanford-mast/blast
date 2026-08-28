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
mod lifecycle;
mod snapshot;
mod store;
mod worker;

use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::info;
use tracing_subscriber::EnvFilter;

use api::handlers::AppState;
use backend::{docker::DockerBackend, hypeman::HypemanBackend, smolvm::SmolvmBackend, Resources, VmBackend};
use config::{BackendConfig, Config};
use snapshot::SnapshotStore;
use store::{RunStore, Store};
use worker::WorkerClient;

#[derive(Parser)]
#[command(name = "blast", about = "Sandbox serving engine")]
struct Cli {
    #[arg(long, env = "BLAST_CONFIG")]
    config: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let mut cfg = Config::load(cli.config.as_deref())?;

    // Runs before backend construction so a zero-config `blast` still comes
    // up: auto-detects (and, if missing, auto-builds/auto-spawns) whatever
    // the configured backend needs, and fills any resolved values (an
    // absolute binary path, a minted hypeman token, ...) back into `cfg`.
    // Needs data_dir to exist first (managed binaries and hypeman's own data
    // dir both live under it).
    tokio::fs::create_dir_all(&cfg.data_dir).await?;
    backend::bootstrap::ensure_backend_ready(&mut cfg.backend, &cfg.data_dir).await?;

    let backend: Arc<dyn VmBackend> = match &cfg.backend {
        BackendConfig::Docker => Arc::new(DockerBackend::new()),
        BackendConfig::Hypeman { endpoint, token, cli_binary } => {
            Arc::new(HypemanBackend::new(endpoint.clone(), token.clone(), cli_binary.clone()))
        }
        BackendConfig::Smolvm { binary } => Arc::new(SmolvmBackend::new(binary.clone(), cfg.data_dir.clone())),
    };

    let snapshots = Arc::new(SnapshotStore::new(&cfg.data_dir));
    let store = Store::new();

    // `{0,0,0}` means no pool configured -- `handle_fork`'s admission
    // control and `lifecycle`'s pressure loop both treat that as unlimited
    // (their `total.x > 0` guards skip straight through), so this single
    // resolution of `worker.resources` is the one thing both consult; see
    // `api::handlers::reserve_pool_capacity` and `lifecycle::pressure_loop`.
    let total_resources = cfg.worker.resources.as_ref().map_or(
        Resources { vcpu: 0, memory_mib: 0, disk_mib: 0 },
        |r| Resources { vcpu: r.vcpu, memory_mib: r.memory_mib, disk_mib: r.disk_mib },
    );

    let state = AppState {
        store: store.clone(),
        runs: RunStore::new(),
        backend: backend.clone(),
        snapshots: snapshots.clone(),
        config: Arc::new(cfg.clone()),
        total_resources: total_resources.clone(),
    };

    // Dirty-sync uploads are only meaningful when a control plane is registered.
    // We share the WorkerClient via RwLock so the upload fn can use the real
    // worker_id (only known after registration) while lifecycle is spawned before
    // the registration future completes.
    let upload_url_fn: Option<lifecycle::UploadUrlFn> =
        if cfg.worker.control_plane_endpoint.is_some() {
            let wc_slot: Arc<RwLock<Option<Arc<WorkerClient>>>> = Arc::new(RwLock::new(None));
            // Store into AppState so worker::start can fill it in after registration.
            // We share a clone with the lifecycle closure.
            let wc_for_lifecycle = wc_slot.clone();
            // worker::start also needs a handle to populate it:
            // We pass wc_slot into AppState via a thread-local approach or by
            // stashing it in a tokio task that runs after registration.
            // Simplest: spawn a task that awaits registration then sets the slot.
            let cfg2 = cfg.clone();
            let state2 = state.clone();
            tokio::spawn(async move {
                match worker::start(cfg2.worker.clone(), state2).await {
                    Ok(Some(wc)) => {
                        *wc_slot.write().await = Some(Arc::new(wc));
                    }
                    Ok(None) => {} // standalone mode
                    Err(e) => tracing::error!("worker registration failed: {e:#}"),
                }
            });

            Some(Arc::new(move |_vm_id: String, blob: String| {
                let wc_ref = wc_for_lifecycle.clone();
                Box::pin(async move {
                    let wc = {
                        let guard = wc_ref.read().await;
                        match guard.as_ref() {
                            Some(w) => Arc::clone(w),
                            None => anyhow::bail!("worker not yet registered; upload URL unavailable"),
                        }
                    };
                    // Call the router's presigned upload URL endpoint.
                    let url = wc.url(&format!("/api/v1/workers/{}/storage/upload-url?blob={}", wc.worker_id, urlencoding::encode(&blob)));
                    let resp = wc.authed(wc.client.get(&url))
                        .send()
                        .await?
                        .error_for_status()?
                        .json::<serde_json::Value>()
                        .await?;
                    resp["url"].as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| anyhow::anyhow!("missing url in presigned response: {resp}"))
                })
            }))
        } else {
            None
        };

    // When upload_url_fn is Some, start() was already spawned above.
    // When None, start it now (standalone mode, no dirty-sync).
    if upload_url_fn.is_none() {
        worker::start(cfg.worker.clone(), state.clone()).await?;
    }

    // Standalone == no control plane registered. Eviction permanently destroys
    // the last local copy of a VM's snapshot, which is only safe when a control
    // plane has (or will have) a durable off-box copy via presigned S3 upload -
    // see lifecycle::transitions::evict_vm.
    let is_standalone = cfg.worker.control_plane_endpoint.is_none();

    lifecycle::spawn(
        &cfg.lifecycle,
        store.clone(),
        &backend,
        snapshots.clone(),
        &total_resources,
        upload_url_fn,
        is_standalone,
    );

    let router = api::router(state)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = format!("0.0.0.0:{}", cfg.port);
    let listener = TcpListener::bind(&addr).await?;
    info!(addr, "blast listening");
    axum::serve(listener, router).await?;

    Ok(())
}
