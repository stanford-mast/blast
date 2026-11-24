//! The API's error envelope (`ErrorResponse` in openapi.json):
//!
//! ```json
//! {"error": {"code": "not_found", "message": "...", "timestamp": "...",
//!            "request_id": "...", "request": {"kind": "matched", "operation_id": "getVm"},
//!            "details": {"resource": "vm"}}}
//! ```
//!
//! Handlers return [`ApiError`], which carries only what the handler knows
//! (status, code, message, details). The [`envelope`] middleware adds the
//! request context and renders the body, and also converts axum's own
//! plain-text rejections (bad JSON, unknown route, wrong method, oversized
//! body) into the same envelope, so every error a client sees has one shape.

use axum::{
    extract::{MatchedPath, Request, State},
    http::{header, HeaderMap, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

use crate::{api::handlers::AppState, worker::records::Activity};

#[derive(Debug, Clone)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    details: Option<Value>,
    /// The API's `recovery` context: what became of the work.
    recovery: Option<Value>,
    activity_recorded: bool,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            details: None,
            recovery: None,
            activity_recorded: false,
        }
    }

    pub const fn status(&self) -> StatusCode {
        self.status
    }

    fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "bad_request", message)
    }

    /// A 400 that names the request field at fault.
    pub fn bad_request_field(field: &str, message: impl Into<String>) -> Self {
        let message = message.into();
        Self::bad_request(message.clone())
            .with_details(json!({ "violations": [{ "field": field, "message": message }] }))
    }

    /// `resource` is one of the API's `ResourceKind`s, e.g. `vm`, `run`, `file`.
    pub fn not_found(resource: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "not_found",
            format!("The {resource} was not found."),
        )
        .with_details(json!({ "resource": resource }))
    }

    pub fn already_exists(resource: &str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, "already_exists", message)
            .with_details(json!({ "resource": resource }))
    }

    /// `feature` is one of the API's `Feature`s; `None` when the refused thing
    /// has no name there.
    pub fn unsupported(feature: Option<&str>, message: impl Into<String>) -> Self {
        let mut details = json!({ "reason": "feature_not_supported" });
        if let Some(feature) = feature {
            details["feature"] = json!(feature);
        }
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_operation",
            message,
        )
        .with_details(details)
    }

    pub fn idempotency_conflict() -> Self {
        Self::new(
            StatusCode::CONFLICT,
            "idempotency_conflict",
            "The idempotency key was used for a different request.",
        )
    }

    /// A key runs its command at most once: the run it started failed.
    pub fn run_failed_under_key(run_id: &str) -> Self {
        let mut error = Self::new(
            StatusCode::CONFLICT,
            "conflict",
            format!("the run this idempotency key started failed: {run_id}"),
        )
        .with_details(json!({ "resource": "run" }));
        error.recovery = Some(json!({ "work": "stopped", "context": { "run_id": run_id } }));
        error
    }

    /// The image a fork named could not be pulled or started: no VM was made.
    pub fn image_fork_failed(message: impl Into<String>) -> Self {
        let mut error = Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "operation_failed",
            message,
        )
        .with_details(json!({ "reason": "image_pull_failed" }));
        error.recovery = Some(json!({ "work": "not_started" }));
        error
    }

    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self::new(StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large", message)
    }

    pub fn capacity_unavailable(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "capacity_unavailable",
            message,
        )
        .with_details(json!({ "scope": "worker" }))
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "unavailable", message)
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "unauthorized", message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", message)
    }

    pub const fn activity_recorded(mut self) -> Self {
        self.activity_recorded = true;
        self
    }

    fn body(&self, method: &Method, matched_path: Option<&str>, request_id: &str) -> Value {
        let request = matched_path
            .and_then(|path| operation_id(method, path))
            .map_or_else(
                || json!({ "kind": "unmatched", "method": method.as_str() }),
                |operation_id| json!({ "kind": "matched", "operation_id": operation_id }),
            );
        let mut error = json!({
            "code": self.code,
            "message": if self.message.is_empty() { self.code } else { &self.message },
            "timestamp": humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string(),
            "request_id": request_id,
            "request": request,
        });
        if let Some(details) = &self.details {
            error["details"] = details.clone();
        }
        if let Some(recovery) = &self.recovery {
            error["recovery"] = recovery.clone();
        }
        json!({ "error": error })
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        tracing::warn!(err = %format!("{e:#}"), "request failed");
        Self::internal(format!("{e:#}"))
    }
}

impl IntoResponse for ApiError {
    /// The body is filled in by [`envelope`], which knows the request.
    fn into_response(self) -> Response {
        let mut res = self.status.into_response();
        res.extensions_mut().insert(self);
        res
    }
}

/// The API's `operationId` for the routes that share one (under `/v1` and
/// `/api/v1` alike); `None` for BLAST-only routes.
fn operation_id(method: &Method, matched_path: &str) -> Option<&'static str> {
    let path = matched_path
        .strip_prefix("/api/v1")
        .or_else(|| matched_path.strip_prefix("/v1"))?;
    Some(match (method.as_str(), path) {
        ("POST", "/fork") => "fork",
        ("GET", "/vms") => "listVms",
        ("GET", "/vms/:vm_id") => "getVm",
        ("DELETE", "/vms/:vm_id") => "deleteVm",
        ("PATCH", "/vms/:vm_id") => "patchVm",
        ("POST", "/vms/:vm_id/runs") => "createRun",
        ("GET", "/vms/:vm_id/runs") => "listRuns",
        ("GET", "/vms/:vm_id/runs/:run_id") => "getRun",
        ("DELETE", "/vms/:vm_id/runs/:run_id") => "cancelRun",
        ("POST", "/vms/:vm_id/sync") => "sync",
        ("GET", "/vms/:vm_id/sessions") => "listSessions",
        ("POST", "/vms/:vm_id/sessions") => "createSession",
        ("GET", "/vms/:vm_id/sessions/:session_id") => "getSession",
        ("PATCH", "/vms/:vm_id/sessions/:session_id") => "patchSession",
        ("DELETE", "/vms/:vm_id/sessions/:session_id") => "deleteSession",
        ("GET", "/vms/:vm_id/sessions/:session_id/pty") => "attachSessionPty",
        ("POST", "/vms/:vm_id/sessions/:session_id/pty-ticket") => "mintSessionPtyTicket",
        ("GET", "/vms/:vm_id/policies") => "getVmPolicies",
        ("PUT", "/vms/:vm_id/policies") => "putVmPolicies",
        ("GET", "/vms/:vm_id/metrics") => "getVmMetrics",
        ("GET", "/vms/:vm_id/mounts") => "listMounts",
        ("POST", "/vms/:vm_id/mounts") => "createMount",
        ("DELETE", "/vms/:vm_id/mounts/:mount_id") => "deleteMount",
        _ => return None,
    })
}

/// Largest rejection body copied into an envelope's `message`.
const MAX_REJECTION_MESSAGE: usize = 4096;

/// Renders every error response as the API's envelope; see the module doc.
pub async fn envelope(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let matched_path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_owned());
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty())
        .map_or_else(|| format!("req_{}", ulid::Ulid::new()), str::to_owned);

    let res = next.run(req).await;
    let status = res.status();
    if !(status.is_client_error() || status.is_server_error()) {
        return res;
    }
    let (mut parts, body) = res.into_parts();
    let error = if let Some(error) = parts.extensions.remove::<ApiError>() {
        error
    } else {
        if is_json(&parts.headers) {
            return Response::from_parts(parts, body);
        }
        let text = axum::body::to_bytes(body, MAX_REJECTION_MESSAGE)
            .await
            .unwrap_or_default();
        rejection_error(
            status,
            &parts.headers,
            String::from_utf8_lossy(&text).trim(),
        )
    };
    if !error.activity_recorded {
        if let Some((action, vm_id)) =
            failed_activity_target(&method, matched_path.as_deref(), &path)
        {
            if let Err(record_error) = state.records.activity(Activity::failed_operation(
                action,
                &request_id,
                vm_id,
                error.status.as_u16(),
                &error.message,
            )) {
                tracing::warn!(err = %record_error, "persist failed-operation activity failed");
            }
        }
    }
    let body = error.body(&method, matched_path.as_deref(), &request_id);
    let mut res = (error.status, Json(body)).into_response();
    for name in [header::ALLOW, header::RETRY_AFTER] {
        if let Some(value) = parts.headers.remove(&name) {
            res.headers_mut().insert(name, value);
        }
    }
    res
}

fn failed_activity_target(
    method: &Method,
    matched_path: Option<&str>,
    path: &str,
) -> Option<(&'static str, Option<String>)> {
    let operation = operation_id(method, matched_path?)?;
    let action = match operation {
        "fork" => "fork",
        "createRun" => "run",
        "sync" => "sync",
        "deleteVm" => "delete",
        _ => return None,
    };
    let parts: Vec<_> = path.split('/').collect();
    let vm_id = parts
        .iter()
        .position(|part| *part == "vms")
        .and_then(|index| parts.get(index + 1))
        .filter(|id| !id.is_empty())
        .map(|id| (*id).to_owned());
    Some((action, vm_id))
}

fn is_json(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"))
}

/// Maps a response produced outside the handlers (axum's extractor
/// rejections and router fallbacks) to the API's code for its status.
fn rejection_error(status: StatusCode, headers: &HeaderMap, message: &str) -> ApiError {
    match status {
        StatusCode::NOT_FOUND => ApiError::not_found("route"),
        StatusCode::METHOD_NOT_ALLOWED => {
            let allowed: Vec<&str> = headers
                .get(header::ALLOW)
                .and_then(|v| v.to_str().ok())
                .map(|v| {
                    v.split(',')
                        .map(str::trim)
                        .filter(|m| !m.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            let error = ApiError::new(
                status,
                "method_not_allowed",
                or(message, "method not allowed"),
            );
            if allowed.is_empty() {
                error
            } else {
                error.with_details(json!({ "allowed_methods": allowed }))
            }
        }
        StatusCode::PAYLOAD_TOO_LARGE => ApiError::new(
            status,
            "payload_too_large",
            or(message, "request body too large"),
        ),
        StatusCode::UNAUTHORIZED => ApiError::unauthorized(or(message, "unauthorized")),
        s if s.is_server_error() => ApiError::internal(or(message, "internal error")),
        // Malformed JSON, a wrong field type, an unknown field, or a missing
        // content type: all a bad request in the API's vocabulary (its 422 is
        // reserved for unsupported operations).
        _ => ApiError::bad_request(or(message, "bad request")),
    }
}

fn or(message: &str, fallback: &str) -> String {
    if message.is_empty() {
        fallback
    } else {
        message
    }
    .to_owned()
}
