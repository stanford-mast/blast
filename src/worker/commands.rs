use std::{sync::Arc, time::Duration};

use tracing::{info, warn};

use crate::api::{
    handlers::AppState,
    types::{WorkerCommand, WorkerCommandResult},
};

use super::WorkerClient;

pub async fn run(wc: Arc<WorkerClient>, state: AppState) {
    loop {
        match poll_command(&wc).await {
            Ok(Some(cmd)) => {
                let result = dispatch(&cmd, &state).await;
                report_result(&wc, &cmd.command_id, result).await;
            }
            Ok(None) => {}
            Err(e) => {
                warn!(err = %e, "command poll failed; retrying in 5s");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }
}

async fn poll_command(wc: &WorkerClient) -> anyhow::Result<Option<WorkerCommand>> {
    let url = wc.url(&format!(
        "/api/v1/workers/{}/commands?timeout_ms=30000",
        wc.worker_id
    ));
    let resp = wc
        .authed(wc.client.get(&url))
        .timeout(Duration::from_secs(35))
        .send()
        .await?;

    if resp.status() == reqwest::StatusCode::NO_CONTENT {
        return Ok(None);
    }
    let cmd = resp.error_for_status()?.json::<WorkerCommand>().await?;
    Ok(Some(cmd))
}

async fn dispatch(cmd: &WorkerCommand, state: &AppState) -> WorkerCommandResult {
    info!(command = %cmd.command, command_id = %cmd.command_id, "dispatching");
    match cmd.command.as_str() {
        "fork" => dispatch_fork(cmd, state).await,
        "run" => dispatch_run(cmd, state).await,
        "delete" => dispatch_delete(cmd, state).await,
        "list_vms" => dispatch_list_vms(state).await,
        "sync" => dispatch_sync(cmd, state).await,
        other => WorkerCommandResult::err(format!("unknown command: {other}")),
    }
}

fn vm_id_param(cmd: &WorkerCommand) -> String {
    cmd.params
        .get("vm_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned()
}

async fn dispatch_fork(cmd: &WorkerCommand, state: &AppState) -> WorkerCommandResult {
    let req = match serde_json::from_value::<crate::api::types::ForkRequest>(cmd.params.clone()) {
        Ok(r) => r,
        Err(e) => return WorkerCommandResult::err(e.to_string()),
    };
    match crate::api::handlers::handle_fork(state, req).await {
        Ok(vm) => WorkerCommandResult { ok: true, result: serde_json::to_value(vm).ok(), error: None },
        Err(e) => WorkerCommandResult::err(e.to_string()),
    }
}

async fn dispatch_run(cmd: &WorkerCommand, state: &AppState) -> WorkerCommandResult {
    let vm_id = vm_id_param(cmd);
    let req = match serde_json::from_value::<crate::api::types::RunRequest>(cmd.params.clone()) {
        Ok(r) => r,
        Err(e) => return WorkerCommandResult::err(e.to_string()),
    };
    match crate::api::handlers::handle_run(state, &vm_id, req).await {
        Ok(run) => WorkerCommandResult { ok: true, result: serde_json::to_value(run).ok(), error: None },
        Err(e) => WorkerCommandResult::err(e.to_string()),
    }
}

async fn dispatch_delete(cmd: &WorkerCommand, state: &AppState) -> WorkerCommandResult {
    let vm_id = vm_id_param(cmd);
    match crate::api::handlers::handle_delete(state, &vm_id).await {
        Ok(_) => WorkerCommandResult { ok: true, result: None, error: None },
        Err(e) => WorkerCommandResult::err(e.to_string()),
    }
}

async fn dispatch_list_vms(state: &AppState) -> WorkerCommandResult {
    let vms = state.store.all().await;
    let cfg = &state.config.worker;
    let list: Vec<crate::api::types::VmObject> = vms
        .into_iter()
        .map(|r| crate::api::types::VmObject {
            vm_id: r.vm_id,
            name: r.name,
            state: crate::api::handlers::api_state_str(&r.state).into(),
            provider: cfg.provider.clone(),
            region: cfg.region.clone().unwrap_or_else(|| "local".into()),
            platform: r.handle.map_or_else(|| crate::backend::host_platform().to_owned(), |h| h.platform),
            resources: crate::api::types::ResourcesOutput {
                vcpu: r.resources.vcpu,
                memory_mib: r.resources.memory_mib,
                disk_mib: r.resources.disk_mib,
            },
        })
        .collect();
    WorkerCommandResult { ok: true, result: serde_json::to_value(list).ok(), error: None }
}

async fn dispatch_sync(cmd: &WorkerCommand, state: &AppState) -> WorkerCommandResult {
    let vm_id = vm_id_param(cmd);
    let req = match serde_json::from_value::<crate::api::types::SyncRequest>(cmd.params.clone()) {
        Ok(r) => r,
        Err(e) => return WorkerCommandResult::err(e.to_string()),
    };
    match crate::api::handlers::handle_sync(state, &vm_id, req).await {
        Ok(resp) => WorkerCommandResult { ok: true, result: serde_json::to_value(resp).ok(), error: None },
        Err(e) => WorkerCommandResult::err(e.to_string()),
    }
}

async fn report_result(wc: &WorkerClient, command_id: &str, result: WorkerCommandResult) {
    let url = wc.url(&format!(
        "/api/v1/workers/{}/commands/{}/result",
        wc.worker_id, command_id
    ));
    if let Err(e) = wc
        .authed(wc.client.post(&url))
        .json(&result)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
    {
        warn!(err = %e, command_id, "failed to report command result");
    }
}
