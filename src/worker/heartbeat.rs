use std::{sync::Arc, time::Duration};

use tokio::time;
use tracing::warn;

use crate::api::{handlers::AppState, types::WorkerHeartbeatRequest};

use super::WorkerClient;

pub async fn run(wc: Arc<WorkerClient>, state: AppState) {
    let mut ticker = time::interval(Duration::from_secs(10));
    loop {
        ticker.tick().await;
        let vms = state.store.all().await;
        let vm_count = u32::try_from(vms.len()).unwrap_or(u32::MAX);
        let in_use = state.store.resources_in_use().await;
        let total = &wc.total_resources;

        let req = WorkerHeartbeatRequest {
            vcpu: total.vcpu,
            memory_mib: total.memory_mib,
            disk_mib: total.disk_mib,
            vcpu_free: total.vcpu.saturating_sub(in_use.vcpu),
            memory_mib_free: total.memory_mib.saturating_sub(in_use.memory_mib),
            disk_mib_free: total.disk_mib.saturating_sub(in_use.disk_mib),
            vm_count,
        };

        let url = wc.url(&format!("/api/v1/workers/{}/heartbeat", wc.worker_id));
        if let Err(e) = wc
            .authed(wc.client.post(&url))
            .json(&req)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
        {
            warn!(err = %e, "heartbeat failed");
        }
    }
}
