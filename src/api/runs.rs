//! Runs and sessions.
//!
//! A session is a label: an index, a starting directory and an environment.
//! Each run is its own shell started there, so nothing a command changes in
//! its shell carries to the next run. Runs of one session execute one at a
//! time, in the order they were accepted.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    Json,
};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::{
    backend::{Captured, Ended, Exec, RunOutput, Stdout, VmBackend, VmHandle},
    store::{new_run_id, RunRecord, RunState, RunStore, SessionRecord, Store, VmRecord},
    worker::records::Records,
};

use super::{
    error::ApiError,
    handlers::{
        begin_mutation, deleted_mid_run, ensure_running_handle, find_vm, lock_vm, rfc3339,
        vm_not_found, AppState,
    },
    types::*,
};

type ApiResult<T> = Result<Json<T>, ApiError>;

/// Tokens that end the runs in flight, by run id.
pub type RunCancels = Arc<Mutex<HashMap<String, CancellationToken>>>;

/// The longest a `createRun` answer waits for its command, and the default.
const MAX_SYNC_WINDOW_SECS: u64 = 300;
const MAX_IDEMPOTENCY_KEY_BYTES: usize = 64;
const IDEMPOTENCY_KEY_HEADER: &str = "idempotency-key";

fn run_not_found() -> ApiError {
    ApiError::not_found("run")
}

fn session_not_found() -> ApiError {
    ApiError::not_found("session")
}

fn session_object(vm: &VmRecord, session: &SessionRecord, running: &HashSet<String>) -> Session {
    Session {
        session_id: session.session_id.clone(),
        session_idx: session.session_idx,
        state: if running.contains(&session.session_id) {
            "running"
        } else {
            "idle"
        },
        cwd: vm.session_cwd(session),
        env: (!session.env.is_empty()).then(|| session.env.clone()),
    }
}

/// The VM's sessions by index, each `running` while one of its runs is.
pub async fn session_objects(s: &AppState, vm: &VmRecord) -> Vec<Session> {
    let running: HashSet<String> = s
        .runs
        .of_vm(&vm.vm_id)
        .await
        .into_iter()
        .filter(|(_, run)| run.state == RunState::Running)
        .filter_map(|(_, run)| run.session_id)
        .collect();
    let mut sessions: Vec<Session> = vm
        .client_sessions()
        .map(|session| session_object(vm, session, &running))
        .collect();
    sessions.sort_by_key(|session| session.session_idx);
    sessions
}

fn persist_session(error: impl std::fmt::Display) -> ApiError {
    ApiError::internal(format!("persist session: {error}"))
}

pub async fn post_session(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
    Json(req): Json<CreateSessionRequest>,
) -> ApiResult<Session> {
    let _mutation = begin_mutation(&s).await?;
    let _vm = lock_vm(&s, &vm_id).await;
    if req.pty == Some(true) || req.command.is_some() || req.cols.is_some() || req.rows.is_some() {
        return Err(ApiError::unsupported(
            Some("persistent_pty"),
            "terminal sessions are not supported",
        ));
    }
    find_vm(&s, &vm_id).await?;
    let session = s
        .store
        .open_session(&vm_id, None, req.cwd, req.env.unwrap_or_default())
        .await
        .map_err(persist_session)?
        .ok_or_else(vm_not_found)?;
    let vm = find_vm(&s, &vm_id).await?;
    Ok(Json(session_object(&vm, &session, &HashSet::new())))
}

pub async fn list_sessions(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
) -> ApiResult<ListSessionsResponse> {
    let vm = find_vm(&s, &vm_id).await?;
    Ok(Json(ListSessionsResponse {
        sessions: session_objects(&s, &vm).await,
        next_cursor: None,
    }))
}

pub async fn get_session(
    State(s): State<AppState>,
    Path((vm_id, session_id)): Path<(String, String)>,
) -> ApiResult<Session> {
    let vm = find_vm(&s, &vm_id).await?;
    session_objects(&s, &vm)
        .await
        .into_iter()
        .find(|session| session.session_id == session_id)
        .map(Json)
        .ok_or_else(session_not_found)
}

/// A session's terminal is not served; a session that does not exist is
/// said not to, before that.
pub async fn session_terminal(
    session: State<AppState>,
    path: Path<(String, String)>,
) -> Result<Json<Session>, ApiError> {
    let _exists = get_session(session, path).await?;
    Err(ApiError::unsupported(
        Some("persistent_pty"),
        "terminals are not supported",
    ))
}

pub async fn delete_session(
    State(s): State<AppState>,
    Path((vm_id, session_id)): Path<(String, String)>,
) -> ApiResult<DeleteResponse> {
    let _mutation = begin_mutation(&s).await?;
    let _vm = lock_vm(&s, &vm_id).await;
    let found = s
        .store
        .remove_session(&vm_id, &session_id)
        .await
        .map_err(persist_session)?;
    if !found {
        return Err(session_not_found());
    }
    Ok(Json(DeleteResponse { deleted: true }))
}

/// The session a run asks for: the one with its id, else the one at its
/// index (index 0 when it names neither), made on first use.
async fn run_session(
    s: &AppState,
    vm: &VmRecord,
    req: &RunRequest,
) -> Result<SessionRecord, ApiError> {
    if let Some(session_id) = &req.session_id {
        return vm
            .sessions
            .get(session_id)
            .cloned()
            .ok_or_else(session_not_found);
    }
    s.store
        .open_session(
            &vm.vm_id,
            Some(req.session_idx.unwrap_or(0)),
            None,
            HashMap::new(),
        )
        .await
        .map_err(persist_session)?
        .ok_or_else(vm_not_found)
}

/// A request's idempotency key and a digest of what it asks for.
fn idempotency(
    headers: &HeaderMap,
    body: &serde_json::Value,
) -> Result<Option<(String, String)>, ApiError> {
    let Some(key) = headers.get(IDEMPOTENCY_KEY_HEADER) else {
        return Ok(None);
    };
    let key = key
        .to_str()
        .ok()
        .filter(|key| {
            (1..=MAX_IDEMPOTENCY_KEY_BYTES).contains(&key.len())
                && key.bytes().all(|byte| (b' '..=b'~').contains(&byte))
        })
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "Idempotency-Key must be 1 to {MAX_IDEMPOTENCY_KEY_BYTES} printable ASCII characters"
            ))
        })?;
    // `serde_json` keeps object keys sorted, so equal requests digest equally.
    let digest = hex::encode(Sha256::digest(body.to_string().as_bytes()));
    Ok(Some((key.to_owned(), digest)))
}

/// Refuses what a run request may ask for that this worker cannot do, and
/// values the API does not allow.
fn check_run_request(req: &RunRequest) -> Result<(), ApiError> {
    if req
        .time_to_background
        .is_some_and(|secs| secs > MAX_SYNC_WINDOW_SECS)
    {
        return Err(ApiError::bad_request_field(
            "time_to_background",
            format!("time_to_background must be at most {MAX_SYNC_WINDOW_SECS}"),
        ));
    }
    if req.signal.is_some() {
        return Err(ApiError::unsupported(
            Some("guest_control"),
            "signals are not supported; cancel the run instead",
        ));
    }
    if req.policies.as_ref().is_some_and(|p| !p.is_null()) {
        return Err(ApiError::unsupported(
            Some("network_policy"),
            "network policies are not supported",
        ));
    }
    if req
        .end_symbol
        .as_deref()
        .is_some_and(|symbol| !matches!(symbol, "auto" | "none"))
    {
        return Err(ApiError::unsupported(
            Some("persistent_pty"),
            "each run is its own shell, so no prompt can end one; omit `end_symbol`",
        ));
    }
    Ok(())
}

pub async fn post_run(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<(StatusCode, Json<RunResponse>), ApiError> {
    let idempotency = idempotency(&headers, &body)?;
    let req: RunRequest =
        serde_json::from_value(body).map_err(|error| ApiError::bad_request(error.to_string()))?;
    handle_run(&s, &vm_id, req, idempotency).await
}

pub async fn handle_run(
    s: &AppState,
    vm_id: &str,
    req: RunRequest,
    idempotency: Option<(String, String)>,
) -> Result<(StatusCode, Json<RunResponse>), ApiError> {
    let _mutation = begin_mutation(s).await?;
    let vm_guard = lock_vm(s, vm_id).await;
    check_run_request(&req)?;
    let vm = find_vm(s, vm_id).await?;
    let Some(command) = req.command.clone().filter(|command| !command.is_empty()) else {
        if req.names_a_run() {
            return Err(ApiError::bad_request(
                "command is required (or use signal parameter)",
            ));
        }
        ensure_running_handle(s, vm_id).await?;
        return Ok((StatusCode::OK, Json(RunResponse::acknowledged())));
    };
    if let Some((key, digest)) = &idempotency {
        if let Some((run_id, run)) = s.runs.keyed(vm_id, key).await {
            drop(vm_guard);
            return replay(s, run_id, run, digest, sync_window(&req)).await;
        }
    }
    let session = run_session(s, &vm, &req).await?;

    let busy = s.store.mark_busy(vm_id).await;
    let handle = match ensure_running_handle(s, vm_id).await {
        Ok(handle) => handle,
        Err(error) => {
            s.store.clear_busy(vm_id, &busy).await;
            return Err(error);
        }
    };

    let mut env = session.env.clone();
    env.extend(req.env.clone().unwrap_or_default());
    let run_id = new_run_id();
    let mut record = RunRecord::started(
        vm_id.to_owned(),
        Some(session.session_id.clone()),
        command.clone(),
    );
    record.idempotency = idempotency;
    if let Err(error) = persist_started_run(s, &run_id, &record).await {
        s.store.clear_busy(vm_id, &busy).await;
        return Err(error);
    }
    // The first run in flight makes the VM `running` to the control plane.
    if !vm.in_use() {
        crate::lifecycle::publish(s, vm_id).await;
    }
    let cancel = CancellationToken::new();
    s.run_cancels
        .lock()
        .expect("run cancels lock")
        .insert(run_id.clone(), cancel.clone());
    // A delete from here on sees this run and ends it.
    drop(vm_guard);

    let task = RunTask {
        backend: s.backend.clone(),
        store: s.store.clone(),
        runs: s.runs.clone(),
        records: s.records.clone(),
        cancels: s.run_cancels.clone(),
        vm_id: vm_id.to_owned(),
        busy,
        run_id: run_id.clone(),
        handle,
        command,
        env,
        cwd: req
            .cwd
            .clone()
            .or(session.cwd)
            .or_else(|| vm.workdir.clone()),
        timeout: req
            .timeout
            .filter(|&secs| secs > 0)
            .map(Duration::from_secs),
        cancel,
    };
    // The command runs on its own task so the answer can return at the sync
    // window without ending it.
    tokio::spawn(task.run());

    let run = wait_for_end(&s.runs, &run_id, sync_window(&req)).await;
    run_response(s, vm_id, run_id, run).await
}

fn sync_window(req: &RunRequest) -> Duration {
    if req.background == Some(true) {
        Duration::ZERO
    } else {
        Duration::from_secs(req.time_to_background.unwrap_or(MAX_SYNC_WINDOW_SECS))
    }
}

/// The answer to a request repeating `run`'s idempotency key: the same run,
/// unless the key was used for another request or its run failed.
async fn replay(
    s: &AppState,
    run_id: String,
    run: RunRecord,
    digest: &str,
    window: Duration,
) -> Result<(StatusCode, Json<RunResponse>), ApiError> {
    if run
        .idempotency
        .as_ref()
        .is_some_and(|(_, original)| original != digest)
    {
        return Err(ApiError::idempotency_conflict());
    }
    if run.state == RunState::Failed {
        return Err(ApiError::run_failed_under_key(&run_id));
    }
    let vm_id = run.vm_id.clone();
    let run = match run.state {
        RunState::Running => wait_for_end(&s.runs, &run_id, window).await,
        _ => Some(run),
    };
    run_response(s, &vm_id, run_id, run).await
}

/// The run once it has ended, or `None` if it still runs after `window` or
/// was forgotten with its VM.
async fn wait_for_end(runs: &RunStore, run_id: &str, window: Duration) -> Option<RunRecord> {
    tokio::time::timeout(window, runs.ended(run_id))
        .await
        .ok()
        .flatten()
}

async fn persist_started_run(
    state: &AppState,
    run_id: &str,
    record: &RunRecord,
) -> Result<(), ApiError> {
    state
        .runs
        .insert(run_id.to_owned(), record.clone())
        .await
        .map_err(|error| ApiError::unavailable(format!("persist run: {error}")))?;
    if let Err(error) = state.records.run(run_id, record) {
        let message = format!("persist run record: {error}");
        let completed_at = SystemTime::now();
        state
            .runs
            .update(run_id, |run| {
                run.state = RunState::Failed;
                run.completed_at = Some(completed_at);
                run.fail_reason = Some(message.clone());
            })
            .await
            .map_err(|persist_error| {
                ApiError::unavailable(format!(
                    "{message}; also failed to persist rejection: {persist_error}"
                ))
            })?;
        return Err(ApiError::unavailable(message));
    }
    Ok(())
}

/// How long a run's answer waits for the run's records to reach the control
/// plane.
const ANSWER_RECORDS_FLUSH: Duration = Duration::from_secs(5);

/// `createRun`'s answer for `run` as it stands at the end of the sync
/// window: `None` when its VM was deleted under it.
async fn run_response(
    s: &AppState,
    vm_id: &str,
    run_id: String,
    run: Option<RunRecord>,
) -> Result<(StatusCode, Json<RunResponse>), ApiError> {
    // A read that follows the answer goes to the control plane's copy of the
    // run; past this wait the record follows the answer instead.
    s.records.flush_vm(vm_id, ANSWER_RECORDS_FLUSH).await;
    let Some(run) = run else {
        return match s.runs.get(&run_id).await {
            Some(run) => Ok((
                StatusCode::ACCEPTED,
                Json(RunResponse::Background {
                    session_id: run.session_id,
                    run_id,
                    state: RunState::Running.as_str(),
                }),
            )),
            None => Err(vm_not_found()),
        };
    };
    if run.state == RunState::Failed {
        // A delete fails the VM's runs before it removes the VM; waiting it
        // out tells a run that failed from one whose VM is gone.
        drop(lock_vm(s, vm_id).await);
        if s.store.get(vm_id).await.is_none() {
            return Err(vm_not_found());
        }
    }
    match run.state {
        RunState::Failed => Err(ApiError::internal(format!(
            "run {run_id} failed: {}",
            run.fail_reason.as_deref().unwrap_or("run failed")
        ))
        .activity_recorded()),
        state => Ok((
            StatusCode::OK,
            Json(RunResponse::Completed {
                session_id: run.session_id,
                run_id: Some(run_id),
                state: state.as_str(),
                stdout: run.stdout.unwrap_or_default(),
                stdout_encoding: encoding_or_utf8(run.stdout_encoding),
                stdout_truncated: run.stdout_truncated,
                stderr: run.stderr.unwrap_or_default(),
                stderr_encoding: encoding_or_utf8(run.stderr_encoding),
                stderr_truncated: run.stderr_truncated,
                exit_code: run.exit_code,
            }),
        )),
    }
}

fn encoding_or_utf8(encoding: Option<&'static str>) -> &'static str {
    encoding.unwrap_or("utf-8")
}

/// One accepted run, executed on its own task.
struct RunTask {
    backend: Arc<dyn VmBackend>,
    store: Store,
    runs: RunStore,
    records: Records,
    cancels: RunCancels,
    vm_id: String,
    busy: String,
    run_id: String,
    handle: VmHandle,
    command: String,
    env: HashMap<String, String>,
    cwd: Option<String>,
    timeout: Option<Duration>,
    cancel: CancellationToken,
}

impl RunTask {
    /// A command stopped at its timeout says so after what it printed to
    /// stderr, as the API's runs do.
    fn say_it_timed_out(&self, mut out: RunOutput) -> RunOutput {
        if let (Ended::TimedOut, Some(timeout)) = (out.ended, self.timeout) {
            if !out.stderr.truncated {
                let said = format!("command timed out after {}ms\n", timeout.as_millis());
                out.stderr.bytes.extend(said.into_bytes());
            }
        }
        out
    }

    /// Runs the command in a shell of its own, and records how it ended.
    async fn run(self) {
        let exec = Exec {
            command: &self.command,
            env: &self.env,
            cwd: self.cwd.as_deref(),
            timeout: self.timeout,
            stdout: Stdout::Capture,
            cancel: &self.cancel,
        };
        let result =
            crate::backend::run_supervised(&*self.backend, &self.handle, exec, &self.run_id)
                .await
                .map(|out| self.say_it_timed_out(out));
        let completed_at = Some(SystemTime::now());
        let end = move |run: &mut RunRecord| match &result {
            Ok(out) => {
                run.state = match out.ended {
                    Ended::Cancelled => RunState::Cancelled,
                    Ended::Exited(_) | Ended::TimedOut => RunState::Completed,
                };
                run.exit_code = out.ended.exit_code();
                (run.stdout, run.stdout_encoding, run.stdout_truncated) =
                    encode_output(&out.stdout);
                (run.stderr, run.stderr_encoding, run.stderr_truncated) =
                    encode_output(&out.stderr);
            }
            Err(error) => {
                run.state = RunState::Failed;
                run.fail_reason = Some(format!("{error:#}"));
            }
        };
        // A run a delete already ended keeps that end, and its record is the
        // delete's to deliver.
        let mut end = Some(end);
        let mut ended_here = false;
        loop {
            let recorded = self
                .runs
                .update(&self.run_id, |run| {
                    if run.state == RunState::Running {
                        if let Some(end) = end.take() {
                            end(run);
                            run.completed_at = completed_at;
                            ended_here = true;
                        }
                    }
                })
                .await;
            let delivered = match recorded {
                Ok(_) if !ended_here => Ok(()),
                Ok(_) => match self.runs.get(&self.run_id).await {
                    Some(run) => self.records.run(&self.run_id, &run),
                    // Only a deleted VM's runs are forgotten.
                    None => Ok(()),
                },
                Err(error) => Err(error),
            };
            match delivered {
                Ok(()) => break,
                Err(error) => {
                    tracing::warn!(run_id = %self.run_id, err = %error, "persist terminal run failed; retrying");
                }
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        self.cancels
            .lock()
            .expect("run cancels lock")
            .remove(&self.run_id);
        self.store.clear_busy(&self.vm_id, &self.busy).await;
        // The last run to end leaves the VM `idle` to the control plane.
        if let Some(vm) = self.store.get(&self.vm_id).await {
            if !vm.in_use() {
                if let Err(error) = self.records.vm(&vm) {
                    tracing::warn!(vm_id = %self.vm_id, err = %error, "persist VM record failed");
                }
            }
        }
    }
}

fn run_object(run_id: String, run: RunRecord) -> Run {
    Run {
        run_id,
        vm_id: run.vm_id,
        session_id: run.session_id,
        command: run.command,
        state: run.state.as_str(),
        started_at: rfc3339(run.started_at),
        completed_at: run.completed_at.map(rfc3339),
        exit_code: run.exit_code,
        fail_reason: run.fail_reason,
        stdout: run.stdout.unwrap_or_default(),
        stdout_encoding: encoding_or_utf8(run.stdout_encoding),
        stdout_truncated: run.stdout_truncated,
        stderr: run.stderr.unwrap_or_default(),
        stderr_encoding: encoding_or_utf8(run.stderr_encoding),
        stderr_truncated: run.stderr_truncated,
    }
}

async fn find_run(s: &AppState, vm_id: &str, run_id: &str) -> Result<RunRecord, ApiError> {
    s.runs
        .get(run_id)
        .await
        .filter(|run| run.vm_id == vm_id)
        .ok_or_else(run_not_found)
}

pub async fn get_run(
    State(s): State<AppState>,
    Path((vm_id, run_id)): Path<(String, String)>,
) -> ApiResult<Run> {
    let run = find_run(&s, &vm_id, &run_id).await?;
    Ok(Json(run_object(run_id, run)))
}

pub async fn list_runs(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
) -> ApiResult<ListRunsResponse> {
    find_vm(&s, &vm_id).await?;
    let mut runs = s.runs.of_vm(&vm_id).await;
    runs.reverse();
    Ok(Json(ListRunsResponse {
        runs: runs
            .into_iter()
            .map(|(run_id, run)| run_object(run_id, run))
            .collect(),
        next_cursor: None,
    }))
}

/// Cancels a run in flight: it is recorded cancelled with the output it has
/// so far, and its command is ended. A run that already ended stays as it
/// ended.
pub async fn cancel_run(
    State(s): State<AppState>,
    Path((vm_id, run_id)): Path<(String, String)>,
) -> ApiResult<CancelRunResponse> {
    let _mutation = begin_mutation(&s).await?;
    if find_run(&s, &vm_id, &run_id).await?.state != RunState::Running {
        return Ok(Json(CancelRunResponse { cancelled: false }));
    }
    let cancel = s
        .run_cancels
        .lock()
        .expect("run cancels lock")
        .get(&run_id)
        .cloned();
    let Some(cancel) = cancel else {
        return Ok(Json(CancelRunResponse { cancelled: false }));
    };
    cancel.cancel();
    // The run's task records the cancellation with the output it collected.
    let ended = wait_for_end(&s.runs, &run_id, CANCEL_WAIT).await;
    match ended {
        Some(run) => Ok(Json(CancelRunResponse {
            cancelled: run.state == RunState::Cancelled,
        })),
        None => Err(ApiError::unavailable(
            "the run has not stopped yet; retry the cancellation",
        )),
    }
}

/// How long a cancellation waits for its command to stop.
const CANCEL_WAIT: Duration = Duration::from_secs(30);

/// Fails every run still in flight on a VM being deleted, as the deletion
/// will cut its command off.
pub async fn fail_active_runs(s: &AppState, vm_id: &str) -> std::io::Result<()> {
    let completed_at = SystemTime::now();
    for (run_id, run) in s.runs.of_vm(vm_id).await {
        if run.state != RunState::Running {
            continue;
        }
        let mut ended = false;
        s.runs
            .update(&run_id, |run| {
                if run.state == RunState::Running {
                    run.state = RunState::Failed;
                    run.completed_at = Some(completed_at);
                    run.exit_code = Some(-1);
                    run.fail_reason = Some(deleted_mid_run(vm_id));
                    ended = true;
                }
            })
            .await?;
        if let Some(run) = s.runs.get(&run_id).await.filter(|_| ended) {
            s.records.run(&run_id, &run)?;
        }
        if let Some(cancel) = s.run_cancels.lock().expect("run cancels lock").get(&run_id) {
            cancel.cancel();
        }
    }
    Ok(())
}

/// Largest run of whole base64 quanta whose encoding fits a record's output.
const BASE64_OUTPUT_LIMIT: usize = crate::backend::OUTPUT_LIMIT / 4 * 3;

/// Command output as text when it is UTF-8, else base64, with the API's
/// `OutputEncoding` name for which and whether the command printed more.
pub fn encode_output(output: &Captured) -> (Option<String>, Option<&'static str>, bool) {
    let text = match std::str::from_utf8(&output.bytes) {
        Ok(text) => Some(text),
        // The cut split a character: what precedes it is the text.
        Err(error) if output.truncated && error.error_len().is_none() => {
            std::str::from_utf8(&output.bytes[..error.valid_up_to()]).ok()
        }
        Err(_) => None,
    };
    if let Some(text) = text {
        return (Some(text.to_owned()), Some("utf-8"), output.truncated);
    }
    let kept = &output.bytes[..output.bytes.len().min(BASE64_OUTPUT_LIMIT)];
    (
        Some(B64.encode(kept)),
        Some("base64"),
        output.truncated || kept.len() < output.bytes.len(),
    )
}
