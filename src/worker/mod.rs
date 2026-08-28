pub mod commands;
pub mod heartbeat;
pub mod register;

use std::sync::Arc;

use anyhow::Result;
use tracing::info;

use crate::{
    api::handlers::AppState,
    api::types::WorkerRegisterRequest,
    backend::Resources,
    config::WorkerConfig,
};

pub struct WorkerClient {
    pub client: reqwest::Client,
    pub endpoint: String,
    pub api_key: String,
    pub worker_id: String,
    /// Configured total resource pool declared at startup.
    pub total_resources: Resources,
}

impl WorkerClient {
    pub fn new(endpoint: &str, api_key: String, worker_id: String, total_resources: Resources) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.trim_end_matches('/').to_owned(),
            api_key,
            worker_id,
            total_resources,
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}/{}", self.endpoint, path.trim_start_matches('/'))
    }

    pub fn authed(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.bearer_auth(&self.api_key)
    }
}

/// Register with the control plane and start heartbeat + command-poll loops.
/// Returns `Ok(Some(WorkerClient))` on success, `Ok(None)` in standalone mode.
pub async fn start(cfg: WorkerConfig, state: AppState) -> Result<Option<WorkerClient>> {
    let (Some(endpoint), Some(api_key)) =
        (cfg.control_plane_endpoint.clone(), cfg.api_key.clone())
    else {
        info!("no control_plane_endpoint configured; running standalone");
        return Ok(None);
    };

    let total = cfg.resources.as_ref().map_or(
        Resources { vcpu: 0, memory_mib: 0, disk_mib: 0 },
        |r| Resources { vcpu: r.vcpu, memory_mib: r.memory_mib, disk_mib: r.disk_mib },
    );

    let reg = WorkerRegisterRequest {
        worker_provider: cfg.provider.clone(),
        worker_region: cfg.region.clone().unwrap_or_else(|| "default".into()),
        platform: state.backend.platform().to_owned(),
        vcpu: total.vcpu,
        memory_mib: total.memory_mib,
        disk_mib: total.disk_mib,
        token: cfg.registration_token.clone(),
    };

    let worker_id = register::register(&endpoint, &api_key, reg).await?;
    info!(%worker_id, "registered with control plane");

    let wc = Arc::new(WorkerClient::new(&endpoint, api_key.clone(), worker_id.clone(), total.clone()));

    let (wc2, state2) = (wc.clone(), state.clone());
    tokio::spawn(async move { heartbeat::run(wc2, state2).await });

    tokio::spawn(async move { commands::run(wc, state).await });

    Ok(Some(WorkerClient::new(&endpoint, api_key, worker_id, total)))
}
