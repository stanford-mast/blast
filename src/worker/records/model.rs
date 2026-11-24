use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    store::{RunRecord, RunState, VmRecord, VmState},
    worker::heartbeat::WorkerResources,
};

const MAX_COMMAND_CHARS: usize = 65_535;
const MAX_FAIL_REASON_CHARS: usize = 4_096;
const MAX_ACTIVITY_OUTPUT_BYTES: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerVmRecord {
    pub state: String,
    pub resources: WorkerResources,
    pub image: Option<String>,
    pub parent_vm_id: Option<String>,
    pub created_at: String,
    pub last_active_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerRunRecord {
    pub state: String,
    pub command: String,
    pub exit_code: Option<i32>,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    /// `utf-8` or `base64`; absent until the run has output.
    #[serde(default)]
    pub stdout_encoding: Option<String>,
    #[serde(default)]
    pub stderr_encoding: Option<String>,
    /// Whether the command printed more than the record carries.
    #[serde(default)]
    pub stdout_truncated: bool,
    #[serde(default)]
    pub stderr_truncated: bool,
    /// Why the worker could not run or finish the command.
    #[serde(default)]
    pub fail_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Activity {
    pub event_id: String,
    pub time_unix_nano: u64,
    pub action_id: String,
    pub action: String,
    pub state: String,
    pub vm_id: Option<String>,
    pub run_id: Option<String>,
    pub exit_code: Option<i32>,
    pub stdout_bytes: Option<usize>,
    pub stderr_bytes: Option<usize>,
    pub requested_at_ms: Option<u64>,
    pub started_at_ms: Option<u64>,
    pub completed_at_ms: Option<u64>,
    pub status_code: u16,
    pub error_message: Option<String>,
    pub response_body_content: Option<String>,
    /// What was asked for: a run's command, a fork's request.
    #[serde(default)]
    pub request_body_content: Option<String>,
}

impl Activity {
    pub fn operation(
        action: impl Into<String>,
        state: impl Into<String>,
        vm_id: Option<String>,
    ) -> Self {
        let event_id = ulid::Ulid::new().to_string();
        let time_unix_nano = now_nanos();
        let state = state.into();
        Self {
            action_id: event_id.clone(),
            event_id,
            time_unix_nano,
            action: action.into(),
            state: state.clone(),
            vm_id,
            run_id: None,
            exit_code: None,
            stdout_bytes: None,
            stderr_bytes: None,
            requested_at_ms: Some(time_unix_nano / 1_000_000),
            started_at_ms: None,
            completed_at_ms: Some(time_unix_nano / 1_000_000),
            status_code: if state == "failed" { 500 } else { 200 },
            error_message: None,
            response_body_content: None,
            request_body_content: None,
        }
    }

    pub fn failed_operation(
        action: impl Into<String>,
        request_id: &str,
        vm_id: Option<String>,
        status_code: u16,
        error_message: &str,
    ) -> Self {
        let time_unix_nano = now_nanos();
        let action = action.into();
        Self {
            event_id: stable_event_id(request_id, &action, "failed"),
            time_unix_nano,
            action_id: request_id.to_owned(),
            action,
            state: "failed".into(),
            vm_id,
            run_id: None,
            exit_code: None,
            stdout_bytes: None,
            stderr_bytes: None,
            requested_at_ms: Some(time_unix_nano / 1_000_000),
            started_at_ms: None,
            completed_at_ms: Some(time_unix_nano / 1_000_000),
            status_code,
            error_message: Some(truncated_bytes(error_message, MAX_ACTIVITY_OUTPUT_BYTES)),
            response_body_content: None,
            request_body_content: None,
        }
    }

    pub(super) fn run(run_id: &str, run: &RunRecord, run_state: RunState) -> Self {
        let state = match run_state {
            RunState::Running => "started",
            RunState::Completed => "completed",
            RunState::Failed => "failed",
            RunState::Cancelled => "cancelled",
        };
        let started_at_ms = epoch_millis(run.started_at);
        let terminal = run_state != RunState::Running;
        let event_time = if terminal {
            run.completed_at.unwrap_or(run.started_at)
        } else {
            run.started_at
        };
        let (
            exit_code,
            stdout_bytes,
            stderr_bytes,
            completed_at_ms,
            error_message,
            response_body_content,
        ) = if terminal {
            (
                    run.exit_code,
                    run.stdout.as_ref().map(String::len),
                    run.stderr.as_ref().map(String::len),
                    run.completed_at.map(epoch_millis),
                    run.fail_reason
                        .as_deref()
                        .map(|message| truncated_bytes(message, MAX_ACTIVITY_OUTPUT_BYTES)),
                    serde_json::to_string(&serde_json::json!({
                        "stdout": run.stdout.as_deref().map(|output| truncated_bytes(output, MAX_ACTIVITY_OUTPUT_BYTES)),
                        "stderr": run.stderr.as_deref().map(|output| truncated_bytes(output, MAX_ACTIVITY_OUTPUT_BYTES)),
                    }))
                    .ok(),
                )
        } else {
            (None, None, None, None, None, None)
        };
        Self {
            event_id: stable_event_id(run_id, "run", state),
            time_unix_nano: epoch_nanos(event_time),
            action_id: run_id.to_owned(),
            action: "run".into(),
            state: state.into(),
            vm_id: Some(run.vm_id.clone()),
            run_id: Some(run_id.to_owned()),
            exit_code,
            stdout_bytes,
            stderr_bytes,
            requested_at_ms: Some(started_at_ms),
            started_at_ms: Some(started_at_ms),
            completed_at_ms,
            status_code: if run_state == RunState::Failed {
                500
            } else {
                200
            },
            error_message,
            response_body_content,
            request_body_content: Some(truncated_bytes(&run.command, MAX_ACTIVITY_OUTPUT_BYTES)),
        }
    }
}

fn epoch_nanos(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |elapsed| {
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
    })
}

fn stable_event_id(action_id: &str, action: &str, state: &str) -> String {
    let digest = Sha256::digest(format!("{action_id}\0{action}\0{state}"));
    format!("evt_{digest:x}")
}

fn epoch_millis(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |elapsed| {
        u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
    })
}

fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
        })
}

fn rfc3339(time: SystemTime) -> String {
    humantime::format_rfc3339_seconds(time).to_string()
}

fn truncated(value: &str, max_chars: usize) -> String {
    value
        .char_indices()
        .nth(max_chars)
        .map_or(value, |(end, _)| &value[..end])
        .to_owned()
}

fn truncated_bytes(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

impl WorkerVmRecord {
    pub fn new(vm: &VmRecord) -> Self {
        let now = SystemTime::now();
        Self {
            state: match vm.state {
                VmState::Pending | VmState::Running if vm.in_use() => "running",
                VmState::Pending | VmState::Running => "idle",
                VmState::Paused => "paused",
                VmState::Suspended => "suspended",
            }
            .into(),
            resources: (&vm.resources).into(),
            image: vm.image.clone(),
            parent_vm_id: vm.parent_vm_id.clone(),
            created_at: rfc3339(vm.created_at),
            // To the millisecond: activity is ordered by it.
            last_active_at: now
                .checked_sub(vm.last_active.elapsed())
                .map(|at| humantime::format_rfc3339_millis(at).to_string()),
        }
    }
}

impl WorkerRunRecord {
    pub fn new(run: &RunRecord) -> Self {
        Self {
            state: run.state.as_str().into(),
            command: truncated(&run.command, MAX_COMMAND_CHARS),
            exit_code: run.exit_code,
            started_at: rfc3339(run.started_at),
            finished_at: run.completed_at.map(rfc3339),
            stdout: run.stdout.clone(),
            stderr: run.stderr.clone(),
            session_id: run.session_id.clone(),
            stdout_encoding: run.stdout_encoding.map(str::to_owned),
            stderr_encoding: run.stderr_encoding.map(str::to_owned),
            stdout_truncated: run.stdout_truncated,
            stderr_truncated: run.stderr_truncated,
            fail_reason: run
                .fail_reason
                .as_deref()
                .map(|reason| truncated(reason, MAX_FAIL_REASON_CHARS)),
        }
    }
}
