pub mod error;
pub mod handlers;
pub mod runs;
pub mod sync;
pub mod types;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::header,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};

use error::ApiError;
use handlers::AppState;

/// The API accepts up to 16 MiB of file content per sync request; base64 and
/// JSON framing push the body past that.
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

/// The API, served under both `/v1` (BLAST's own paths) and `/api/v1` (the
/// paths the control plane forwards through the tunnel). With `local_token`
/// set, every request except the liveness probe must carry it as a bearer
/// token; the tunnel's router is built without one.
pub fn router(state: AppState, local_token: Option<Arc<str>>) -> Router {
    router_with(state, local_token, Router::new())
}

/// [`router`] plus `extra` routes, behind the same token.
pub fn router_with(
    state: AppState,
    local_token: Option<Arc<str>>,
    extra: Router<AppState>,
) -> Router {
    let api = Router::new()
        .route("/regions", get(handlers::get_regions))
        .route("/fork", post(handlers::post_fork))
        .route("/vms", get(handlers::list_vms))
        .route(
            "/vms/:vm_id",
            get(handlers::get_vm)
                .delete(handlers::delete_vm)
                .patch(handlers::patch_vm),
        )
        .route(
            "/vms/:vm_id/runs",
            post(runs::post_run).get(runs::list_runs),
        )
        .route("/vms/:vm_id/prepare-claim", post(handlers::prepare_claim))
        .route(
            "/vms/:vm_id/claim",
            post(handlers::commit_claim).delete(handlers::abandon_claim),
        )
        .route(
            "/vms/:vm_id/runs/:run_id",
            get(runs::get_run).delete(runs::cancel_run),
        )
        .route(
            "/vms/:vm_id/sessions",
            post(runs::post_session).get(runs::list_sessions),
        )
        .route(
            "/vms/:vm_id/sessions/:session_id",
            get(runs::get_session)
                .delete(runs::delete_session)
                .patch(runs::session_terminal),
        )
        .route(
            "/vms/:vm_id/sessions/:session_id/pty",
            get(runs::session_terminal),
        )
        .route(
            "/vms/:vm_id/sessions/:session_id/pty-ticket",
            post(runs::session_terminal),
        )
        .route(
            "/vms/:vm_id/policies",
            get(unserved::network_policy).put(unserved::network_policy),
        )
        .route("/vms/:vm_id/metrics", get(unserved::metrics))
        .route(
            "/vms/:vm_id/mounts",
            get(unserved::filesystem).post(unserved::filesystem),
        )
        .route(
            "/vms/:vm_id/mounts/:mount_id",
            axum::routing::delete(unserved::filesystem),
        )
        .route("/vms/:vm_id/sync", post(sync::post_sync))
        .route("/live", get(handlers::get_live));

    finish_router(api, state, local_token, extra, true)
}

/// Read-only diagnostics for a remotely controlled worker's loopback listener.
pub fn diagnostics_router(
    state: AppState,
    local_token: Option<Arc<str>>,
    extra: Router<AppState>,
) -> Router {
    let api = if local_token.is_some() {
        Router::new()
            .route("/regions", get(handlers::get_regions))
            .route("/vms", get(handlers::list_vms))
            .route("/vms/:vm_id", get(handlers::get_vm))
            .route("/vms/:vm_id/runs/:run_id", get(runs::get_run))
            .route("/vms/:vm_id/sessions", get(runs::list_sessions))
            .route("/live", get(handlers::get_live))
    } else {
        Router::new().route("/live", get(handlers::get_live))
    };
    let expose_metrics = local_token.is_some();
    finish_router(api, state, local_token, extra, expose_metrics)
}

fn finish_router(
    api: Router<AppState>,
    state: AppState,
    local_token: Option<Arc<str>>,
    extra: Router<AppState>,
    expose_metrics: bool,
) -> Router {
    let mut router = Router::new()
        .nest("/v1", api.clone())
        .nest("/api/v1", api)
        .merge(extra)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES));
    if expose_metrics {
        router = router.route("/metrics", get(handlers::get_metrics));
    }
    if let Some(token) = local_token {
        router = router.layer(middleware::from_fn_with_state(token, require_token));
    }
    router
        .layer(middleware::from_fn_with_state(
            state.clone(),
            error::envelope,
        ))
        .with_state(state)
}

/// The API's operations this worker does not serve, each refused by name
/// rather than left to a route-not-found.
mod unserved {
    use super::error::ApiError;

    pub async fn network_policy() -> ApiError {
        ApiError::unsupported(Some("network_policy"), "network policies are not supported")
    }

    pub async fn metrics() -> ApiError {
        ApiError::unsupported(Some("vm_metrics"), "VM metrics are not supported")
    }

    pub async fn filesystem() -> ApiError {
        ApiError::unsupported(
            Some("shared_filesystem"),
            "shared filesystems are not supported",
        )
    }
}

async fn require_token(State(token): State<Arc<str>>, req: Request, next: Next) -> Response {
    let path = req.uri().path();
    if path == "/v1/live" || path == "/api/v1/live" {
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match presented {
        Some(presented) if constant_time_eq(presented.as_bytes(), token.as_bytes()) => {
            next.run(req).await
        }
        _ => ApiError::unauthorized("missing or invalid bearer token").into_response(),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}
