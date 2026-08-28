use anyhow::{Context, Result};
use reqwest::Client;

use crate::api::types::{WorkerRegisterRequest, WorkerRegisterResponse};

pub async fn register(
    endpoint: &str,
    api_key: &str,
    req: WorkerRegisterRequest,
) -> Result<String> {
    let url = format!("{}/api/v1/workers/register", endpoint.trim_end_matches('/'));
    let resp = Client::new()
        .post(&url)
        .bearer_auth(api_key)
        .json(&req)
        .send()
        .await
        .context("worker register request")?
        .error_for_status()
        .context("worker register status")?
        .json::<WorkerRegisterResponse>()
        .await
        .context("worker register parse")?;
    Ok(resp.worker_id)
}
