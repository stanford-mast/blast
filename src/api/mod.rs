pub mod handlers;
pub mod types;

use axum::{
    routing::{delete, get, post},
    Router,
};

use handlers::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/regions", get(handlers::get_regions))
        .route("/v1/fork", post(handlers::post_fork))
        .route("/v1/vms", get(handlers::list_vms))
        .route("/v1/vms/:vm_id", get(handlers::get_vm).delete(handlers::delete_vm))
        .route("/v1/vms/:vm_id/runs", post(handlers::post_run))
        .route("/v1/vms/:vm_id/runs/:run_id", get(handlers::get_run))
        .route("/v1/vms/:vm_id/sessions", post(handlers::post_session))
        .route("/v1/vms/:vm_id/sessions", get(handlers::list_sessions))
        .route(
            "/v1/vms/:vm_id/sessions/:session_id",
            delete(handlers::delete_session),
        )
        .route("/v1/vms/:vm_id/sync", post(handlers::post_sync))
        .route("/v1/live", get(handlers::get_live))
        // Bare `/metrics`, not `/v1/metrics`: every Prometheus-compatible
        // scraper defaults to this path, and there's no `/v1/`-versioned API
        // contract at stake here the way there is for the VM-management
        // routes above.
        .route("/metrics", get(handlers::get_metrics))
        .with_state(state)
}
