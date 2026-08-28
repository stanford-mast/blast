use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use base64::{engine::general_purpose::STANDARD as B64, Engine};

use crate::{
    backend::{Resources, VmBackend},
    config::Config,
    snapshot::SnapshotStore,
    store::{
        new_run_id, new_session_id, new_vm_id, RunRecord, RunState, RunStore, SessionRecord,
        SessionState, Store, VmRecord, VmState,
    },
};

use super::types::*;

#[derive(Clone)]
pub struct AppState {
    pub store: Store,
    pub runs: RunStore,
    pub backend: Arc<dyn VmBackend>,
    pub snapshots: Arc<SnapshotStore>,
    pub config: Arc<Config>,
    /// The same resolved pool total `lifecycle::spawn` drives its pressure
    /// loop against (see `main.rs`) -- `{0,0,0}` means no pool configured,
    /// matching `pressure_loop`'s own `total.x > 0` guards. Kept as one
    /// canonical resolution of `config.worker.resources` rather than a
    /// second copy derived independently in the fork handler.
    pub total_resources: Resources,
}

type ApiResult<T> = Result<Json<T>, (StatusCode, Json<serde_json::Value>)>;

fn api_err(code: StatusCode, msg: impl std::fmt::Display) -> (StatusCode, Json<serde_json::Value>) {
    (code, Json(serde_json::json!({ "code": code.as_u16(), "message": msg.to_string() })))
}

fn internal(e: anyhow::Error) -> (StatusCode, Json<serde_json::Value>) {
    api_err(StatusCode::INTERNAL_SERVER_ERROR, e)
}

fn not_found(e: anyhow::Error) -> (StatusCode, Json<serde_json::Value>) {
    api_err(StatusCode::NOT_FOUND, e)
}

// ── Regions ───────────────────────────────────────────────────────────────────

pub async fn get_regions(State(s): State<AppState>) -> ApiResult<ListRegionsResponse> {
    let cfg = &s.config.worker;
    let resources = cfg.resources.as_ref();
    Ok(Json(ListRegionsResponse {
        regions: vec![RegionEntry {
            provider: cfg.provider.clone(),
            region: cfg.region.clone().unwrap_or_else(|| "local".into()),
            endpoint: format!("http://localhost:{}", s.config.port),
            platform: s.backend.platform().to_owned(),
            vcpu: resources.map_or(0, |r| r.vcpu),
            memory_mib: resources.map_or(0, |r| r.memory_mib),
            disk_mib: resources.map_or(0, |r| r.disk_mib),
        }],
    }))
}

// ── Fork ──────────────────────────────────────────────────────────────────────

pub async fn post_fork(
    State(s): State<AppState>,
    Json(req): Json<ForkRequest>,
) -> ApiResult<VmObject> {
    handle_fork(&s, req).await.map(Json).map_err(internal)
}

pub async fn handle_fork(s: &AppState, req: ForkRequest) -> anyhow::Result<VmObject> {
    let explicit = req.resources.as_ref().map(|r| Resources {
        vcpu: r.vcpu.unwrap_or(2),
        memory_mib: r.memory_mib.unwrap_or(2048),
        disk_mib: r.disk_mib.unwrap_or(10240),
    });
    // A VM-fork always inherits the source's actual shape: none of BLAST's
    // backends can honor a resize on fork-from-a-running-VM. SmolVM's
    // live-checkpoint restore rejects it outright at the CLI (confirmed:
    // 'topology-changing create flags are not supported', and even a
    // metadata-only `machine update` afterward still fails to boot --
    // EINVAL, a libkrun constraint, not a smolvm one). Docker and Hypeman
    // don't reject it -- they silently IGNORE the request and inherit the
    // source's size anyway (confirmed the hard way: a fork requesting 4
    // vcpu/4096MiB against a 1 vcpu/1024MiB hypeman source came back
    // reporting 4/4096 while `nproc` inside it still said 1). So this is
    // checked once here, backend-agnostically, against BLAST's own source
    // record -- rather than trusting each backend to either reject or
    // honor it correctly on its own.
    let source = match resolve_source_vm_id(s, &req).await {
        Some(src_id) => s.store.get(&src_id).await.map(|src| (src_id, src)),
        None => None,
    };
    let resources = match (&explicit, &source) {
        (Some(want), Some((src_id, src))) if *want != src.resources => {
            anyhow::bail!(
                "cannot resize on fork from a running VM: source {src_id} is {}vcpu/{}MiB/{}MiB disk, requested {}vcpu/{}MiB/{}MiB disk. A VM-fork always inherits its source's exact resources; omit `resources` to inherit it, or fork from an image instead if a specific size is required.",
                src.resources.vcpu, src.resources.memory_mib, src.resources.disk_mib,
                want.vcpu, want.memory_mib, want.disk_mib
            );
        }
        (Some(want), _) => want.clone(),
        (None, Some((_, src))) => src.resources.clone(),
        (None, None) => Resources { vcpu: 2, memory_mib: 2048, disk_mib: 10240 },
    };

    let vm_id = new_vm_id();
    let name = req.name.clone();

    // Local admission control: when this worker declares a pool
    // (`[worker].resources`), a fork must actually fit in it. No pool
    // configured keeps pre-existing unlimited behavior (returns `None`).
    // See `admission` for the single wait/queue/fail-fast implementation
    // shared with the resume path in `ensure_running_handle`.
    let pool = crate::admission::acquire_for_fork(
        &s.store,
        &s.total_resources,
        s.config.worker.admission_queue_secs,
        &vm_id,
        name.clone(),
        &resources,
    )
    .await?;

    let fork_result = resolve_fork_handle(s, req, &resources).await;
    let handle = match fork_result {
        Ok(h) => h,
        Err(e) => {
            crate::admission::release_failed_fork(&s.store, &vm_id, pool.as_ref()).await;
            return Err(e);
        }
    };

    if pool.is_some() {
        // Promote the `Pending` placeholder reserved above.
        s.store
            .update(&vm_id, |r| {
                r.state = VmState::Running;
                r.handle = Some(handle.clone());
            })
            .await;
    } else {
        let record = VmRecord {
            vm_id: vm_id.clone(),
            name: name.clone(),
            state: VmState::Running,
            resources: resources.clone(),
            snapshot_dir: None,
            handle: Some(handle.clone()),
            last_active: std::time::Instant::now(),
            idle_since: None,
            sessions: HashMap::new(),
        };
        s.store.insert(record).await;
    }

    let cfg = &s.config.worker;
    Ok(VmObject {
        vm_id,
        name,
        state: "running".into(),
        provider: cfg.provider.clone(),
        region: cfg.region.clone().unwrap_or_else(|| "local".into()),
        platform: handle.platform,
        resources: ResourcesOutput {
            vcpu: resources.vcpu,
            memory_mib: resources.memory_mib,
            disk_mib: resources.disk_mib,
        },
    })
}

/// Resolves a fork request to a live backend handle: either booting fresh
/// from an OCI image, or snapshotting/reading a source VM and forking from
/// that. Split out of `handle_fork` so admission control can wrap this one
/// fallible step and release the reservation on any error path here.

/// Resolve source by id first, then scan by name if only a name was
/// provided. Shared between `handle_fork`'s resources-inheritance lookup and
/// `resolve_fork_handle`'s snapshot lookup so both agree on the same VM.
async fn resolve_source_vm_id(s: &AppState, req: &ForkRequest) -> Option<String> {
    if let Some(ref id) = req.source_vm_id {
        Some(id.clone())
    } else if let Some(ref name) = req.source_vm_name {
        s.store
            .all()
            .await
            .into_iter()
            .find(|vm| vm.name.as_deref() == Some(name.as_str()))
            .map(|vm| vm.vm_id)
    } else {
        None
    }
}

async fn resolve_fork_handle(
    s: &AppState,
    req: ForkRequest,
    resources: &Resources,
) -> anyhow::Result<crate::backend::VmHandle> {
    if let Some(image) = req.image {
        return s.backend.fork_image(&image, resources, req.registry_auth.as_ref()).await;
    }
    let src_id = resolve_source_vm_id(s, &req)
        .await
        .ok_or_else(|| anyhow::anyhow!("one of: image, source_vm_id, source_vm_name required"))?;
    let src = s
        .store
        .get(&src_id)
        .await
        .ok_or_else(|| anyhow::anyhow!("source VM not found: {src_id}"))?;
    // If the source VM is running, snapshot to a unique per-fork directory so
    // concurrent forks from the same source don't collide on the same snap_dir.
    let snap_dir = if let Some(ref h) = src.handle {
        let fork_snap_dir = s.snapshots.snap_dir(&format!("{}_{}", src_id, ulid::Ulid::new()));
        tokio::fs::create_dir_all(&fork_snap_dir).await?;

        // Mark the source VM busy for the snapshot's duration so the
        // background TTL loop's pause_vm()/suspend_vm() can't race a
        // transition against this same backend handle mid-snapshot --
        // observed for real as a 409 "invalid state for standby" at the
        // hypeman level when suspend_vm() fired on a VM this snapshot call
        // was still using. Reuses the existing session guard
        // (transitions::safe_to_transition already refuses to touch a VM
        // with any session Running) rather than a new field: an in-flight
        // fork-from-this-source is exactly the same "don't touch this VM's
        // backend state right now" as a running session.
        let guard_id = format!("fork_guard_{}", ulid::Ulid::new());
        s.store
            .update(&src_id, |r| {
                r.sessions.insert(
                    guard_id.clone(),
                    SessionRecord {
                        session_id: guard_id.clone(),
                        session_idx: u32::MAX,
                        cwd: "/".into(),
                        env: HashMap::new(),
                        state: SessionState::Running,
                    },
                );
            })
            .await;
        let snapshot_result = s.backend.snapshot(h, &fork_snap_dir).await;
        s.store.update(&src_id, |r| { r.sessions.remove(&guard_id); }).await;
        snapshot_result?;

        fork_snap_dir
    } else {
        // VM is suspended, read from its existing snapshot.
        src.snapshot_dir.unwrap_or_else(|| s.snapshots.snap_dir(&src_id))
    };
    s.backend.fork_snapshot(&snap_dir, resources).await
}

// ── Delete VM ─────────────────────────────────────────────────────────────────

pub async fn delete_vm(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
) -> ApiResult<DeleteResponse> {
    handle_delete(&s, &vm_id).await.map(Json).map_err(not_found)
}

pub async fn handle_delete(s: &AppState, vm_id: &str) -> anyhow::Result<DeleteResponse> {
    let rec = s
        .store
        .remove(vm_id)
        .await
        .ok_or_else(|| anyhow::anyhow!("VM not found: {vm_id}"))?;
    if let Some(handle) = rec.handle {
        s.backend.delete(&handle).await.ok();
    }
    // Deleting frees whatever this VM held (vcpu+memory+disk, or just disk
    // if it was suspended) -- wake anything parked in `reserve_pool_capacity`.
    s.store.notify_capacity_freed();
    Ok(DeleteResponse { deleted: true })
}

// ── List / Get VMs ───────────────────────────────────────────────────────────

/// The public API's state vocabulary is just "running"/"idle", matching
/// arker's own control-plane API -- BLAST's richer internal lifecycle
/// (Pending/Running/Paused/Suspended) still drives admission control and
/// the pressure loop, but that distinction stays internal, not part of the
/// public contract.
pub const fn api_state_str(state: &VmState) -> &'static str {
    match state {
        VmState::Pending | VmState::Running => "running",
        VmState::Paused | VmState::Suspended => "idle",
    }
}

pub async fn list_vms(State(s): State<AppState>) -> ApiResult<ListVmsResponse> {
    let cfg = &s.config.worker;
    let records = s.store.all().await;
    let vms = records
        .into_iter()
        .map(|rec| {
            let state = api_state_str(&rec.state);
            let platform = rec.handle.as_ref().map_or_else(
                || crate::backend::host_platform().to_owned(),
                |h| h.platform.clone(),
            );
            VmObject {
                vm_id: rec.vm_id,
                name: rec.name,
                state: state.into(),
                provider: cfg.provider.clone(),
                region: cfg.region.clone().unwrap_or_else(|| "local".into()),
                platform,
                resources: ResourcesOutput {
                    vcpu: rec.resources.vcpu,
                    memory_mib: rec.resources.memory_mib,
                    disk_mib: rec.resources.disk_mib,
                },
            }
        })
        .collect();
    Ok(Json(ListVmsResponse { vms }))
}

pub async fn get_vm(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
) -> ApiResult<VmObject> {
    let rec = s
        .store
        .get(&vm_id)
        .await
        .ok_or_else(|| api_err(StatusCode::NOT_FOUND, "VM not found"))?;
    let cfg = &s.config.worker;
    let state = api_state_str(&rec.state);
    let platform = rec.handle.as_ref().map_or_else(
        || crate::backend::host_platform().to_owned(),
        |h| h.platform.clone(),
    );
    Ok(Json(VmObject {
        vm_id: rec.vm_id,
        name: rec.name,
        state: state.into(),
        provider: cfg.provider.clone(),
        region: cfg.region.clone().unwrap_or_else(|| "local".into()),
        platform,
        resources: ResourcesOutput {
            vcpu: rec.resources.vcpu,
            memory_mib: rec.resources.memory_mib,
            disk_mib: rec.resources.disk_mib,
        },
    }))
}

// ── Run ───────────────────────────────────────────────────────────────────────

pub async fn post_run(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
    Json(req): Json<RunRequest>,
) -> ApiResult<RunResponse> {
    handle_run(&s, &vm_id, req).await.map(Json).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("VM not found") || msg.contains("session not found") {
            not_found(e)
        } else {
            internal(e)
        }
    })
}

/// Ensures the VM has a live backend handle before running a command,
/// auto-resuming from pause or from a suspended snapshot as needed.
async fn ensure_running_handle(
    s: &AppState,
    vm_id: &str,
    vm: VmRecord,
) -> anyhow::Result<crate::backend::VmHandle> {
    let original_state = vm.state.clone();
    let admit = || {
        crate::admission::acquire_for_resume(
            &s.store,
            &s.total_resources,
            s.config.worker.admission_queue_secs,
            vm_id,
            &vm.resources,
        )
    };
    match (vm.state, vm.handle) {
        (VmState::Paused, Some(h)) => {
            // Auto-resume: re-admit the vcpu this unpause re-acquires
            // against the pool before touching the backend -- the other
            // admission-controlled moment alongside fork, sharing the same
            // wait/queue/fail-fast implementation (see `admission`).
            admit().await?;
            match s.backend.unpause(&h).await {
                Ok(()) => {
                    // Only now -- the backend call actually succeeded -- is
                    // it safe to mark this Running; until this point the
                    // reservation left it `Pending` specifically so the
                    // pressure loop couldn't grab it out from under an
                    // in-flight unpause (see `Store::try_reserve_resume`).
                    s.store
                        .update(vm_id, |r| {
                            r.state = VmState::Running;
                            r.last_active = std::time::Instant::now();
                            r.idle_since = None;
                        })
                        .await;
                    Ok(h)
                }
                Err(e) => {
                    s.store.update(vm_id, |r| r.state = original_state).await;
                    s.store.notify_capacity_freed();
                    Err(e)
                }
            }
        }
        (_, Some(h)) => Ok(h),
        (_, None) => {
            let snap = vm
                .snapshot_dir
                .clone()
                .ok_or_else(|| anyhow::anyhow!("VM is idle but has no snapshot"))?;
            admit().await?;
            match s.backend.resume(&snap, &vm.resources).await {
                Ok(h) => {
                    // Same reasoning as the unpause branch above: the
                    // reservation left this `Pending`, invisible to the
                    // pressure loop, until the backend call actually lands.
                    s.store
                        .update(vm_id, |r| {
                            r.state = VmState::Running;
                            r.handle = Some(h.clone());
                            r.last_active = std::time::Instant::now();
                            r.idle_since = None;
                        })
                        .await;
                    Ok(h)
                }
                Err(e) => {
                    s.store
                        .update(vm_id, |r| {
                            r.state = original_state;
                            r.handle = None;
                        })
                        .await;
                    s.store.notify_capacity_freed();
                    Err(e)
                }
            }
        }
    }
}

pub async fn handle_run(
    s: &AppState,
    vm_id: &str,
    req: RunRequest,
) -> anyhow::Result<RunResponse> {
    let command = req
        .command
        .ok_or_else(|| anyhow::anyhow!("`command` required"))?;

    let vm = s
        .store
        .get(vm_id)
        .await
        .ok_or_else(|| anyhow::anyhow!("VM not found: {vm_id}"))?;

    // Validate session_id if provided.
    if let Some(ref sid) = req.session_id {
        if !vm.sessions.contains_key(sid) {
            anyhow::bail!("session not found: {sid}");
        }
    }
    let session_id = req
        .session_id
        .unwrap_or_else(|| format!("sess_default_{vm_id}"));

    let handle = ensure_running_handle(s, vm_id, vm).await?;

    let env = req.env.unwrap_or_default();
    let cwd = req.cwd.unwrap_or_else(|| "/".into());
    let timeout = Duration::from_secs(req.timeout.unwrap_or(300));
    let sync_window = Duration::from_secs(req.time_to_background.unwrap_or(300));
    let run_id = new_run_id();

    // Mark session as Running while the command executes.
    s.store
        .update(vm_id, |r| {
            if let Some(sess) = r.sessions.get_mut(&session_id) {
                sess.state = SessionState::Running;
            }
        })
        .await;

    s.runs
        .insert(
            run_id.clone(),
            RunRecord {
                state: RunState::Running,
                stdout: None,
                stderr: None,
                stdout_encoding: None,
                stderr_encoding: None,
                exit_code: None,
                fail_reason: None,
            },
        )
        .await;

    // Run the backend command in the background so the HTTP response can be
    // returned early (`time_to_background`) without aborting the command.
    // The run store is the source of truth for a caller that ends up
    // polling; the oneshot channel is only a fast path for callers whose
    // command finishes inside the sync window.
    let (tx, rx) = tokio::sync::oneshot::channel();
    let run_id_task = run_id.clone();
    let ctx = RunTaskCtx {
        backend: s.backend.clone(),
        store: s.store.clone(),
        runs: s.runs.clone(),
        vm_id: vm_id.to_owned(),
        session_id: session_id.clone(),
        run_id: run_id_task,
    };

    tokio::spawn(async move {
        let response = run_and_record(ctx, handle, command, env, cwd, timeout).await;
        // A send error just means the sync wait already elapsed and the
        // caller moved on to polling the run store, nothing to do.
        let _ = tx.send(response);
    });

    // Race the command's completion against the sync window. `time_to_background:
    // Some(0)` collapses this to a near-zero timeout, so the response returns
    // immediately with a pollable run_id without waiting on the command at all.
    match tokio::time::timeout(sync_window, rx).await {
        Ok(Ok(Ok(resp))) => Ok(resp),
        Ok(Ok(Err(fail_reason))) => Ok(RunResponse {
            run_id,
            state: RunState::Failed.as_str().into(),
            stdout: None,
            stderr: None,
            stdout_encoding: None,
            stderr_encoding: None,
            exit_code: None,
            fail_reason: Some(fail_reason),
        }),
        Ok(Err(_)) | Err(_) => Ok(RunResponse {
            run_id,
            state: RunState::Running.as_str().into(),
            stdout: None,
            stderr: None,
            stdout_encoding: None,
            stderr_encoding: None,
            exit_code: None,
            fail_reason: None,
        }),
    }
}

/// Everything the spawned run task needs, bundled to keep `handle_run` short.
struct RunTaskCtx {
    backend: Arc<dyn VmBackend>,
    store: Store,
    runs: RunStore,
    vm_id: String,
    session_id: String,
    run_id: String,
}

/// Executes the backend command to completion, updates the run store and
/// session bookkeeping, and returns the eventual `RunResponse` (or a
/// `fail_reason` string on execution failure) for the oneshot fast path.
async fn run_and_record(
    ctx: RunTaskCtx,
    handle: crate::backend::VmHandle,
    command: String,
    env: HashMap<String, String>,
    cwd: String,
    timeout: Duration,
) -> Result<RunResponse, String> {
    let result = ctx
        .backend
        .run(&handle, &command, &ctx.session_id, &env, &cwd, timeout)
        .await;

    // Reset session state and bump last_active regardless of run outcome.
    ctx.store
        .update(&ctx.vm_id, |r| {
            if let Some(sess) = r.sessions.get_mut(&ctx.session_id) {
                sess.state = SessionState::Idle;
            }
            r.last_active = std::time::Instant::now();
        })
        .await;

    match result {
        Ok(out) => {
            let (stdout, stdout_enc) = encode_output(&out.stdout);
            let (stderr, stderr_enc) = encode_output(&out.stderr);
            ctx.runs
                .update(&ctx.run_id, |rec| {
                    rec.state = RunState::Completed;
                    rec.stdout = Some(stdout.clone());
                    rec.stderr = Some(stderr.clone());
                    rec.stdout_encoding = Some(stdout_enc.to_owned());
                    rec.stderr_encoding = Some(stderr_enc.to_owned());
                    rec.exit_code = Some(out.exit_code);
                })
                .await;
            Ok(RunResponse {
                run_id: ctx.run_id,
                state: RunState::Completed.as_str().into(),
                stdout: Some(stdout),
                stderr: Some(stderr),
                stdout_encoding: Some(stdout_enc.to_owned()),
                stderr_encoding: Some(stderr_enc.to_owned()),
                exit_code: Some(out.exit_code),
                fail_reason: None,
            })
        }
        Err(e) => {
            let fail_reason = e.to_string();
            ctx.runs
                .update(&ctx.run_id, |rec| {
                    rec.state = RunState::Failed;
                    rec.fail_reason = Some(fail_reason.clone());
                })
                .await;
            Err(fail_reason)
        }
    }
}

// ── Poll run ──────────────────────────────────────────────────────────────────

pub async fn get_run(
    State(s): State<AppState>,
    Path((_vm_id, run_id)): Path<(String, String)>,
) -> ApiResult<RunResponse> {
    let rec = s
        .runs
        .get(&run_id)
        .await
        .ok_or_else(|| api_err(StatusCode::NOT_FOUND, format!("run not found: {run_id}")))?;
    Ok(Json(RunResponse {
        run_id,
        state: rec.state.as_str().into(),
        stdout: rec.stdout,
        stderr: rec.stderr,
        stdout_encoding: rec.stdout_encoding,
        stderr_encoding: rec.stderr_encoding,
        exit_code: rec.exit_code,
        fail_reason: rec.fail_reason,
    }))
}

// ── Sessions ──────────────────────────────────────────────────────────────────

pub async fn post_session(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
    Json(req): Json<CreateSessionRequest>,
) -> ApiResult<SessionObject> {
    if s.store.get(&vm_id).await.is_none() {
        return Err(api_err(StatusCode::NOT_FOUND, "VM not found"));
    }
    let session_id = new_session_id();
    let session_idx = {
        let mut idx = 0u32;
        s.store
            .update(&vm_id, |r| {
                idx = u32::try_from(r.sessions.len()).unwrap_or(u32::MAX);
                r.sessions.insert(
                    session_id.clone(),
                    SessionRecord {
                        session_id: session_id.clone(),
                        session_idx: idx,
                        cwd: req.cwd.clone().unwrap_or_else(|| "/".into()),
                        env: req.env.clone().unwrap_or_default(),
                        state: SessionState::Idle,
                    },
                );
            })
            .await;
        idx
    };
    Ok(Json(SessionObject {
        session_id,
        session_idx,
        state: "idle".into(),
        cwd: req.cwd.unwrap_or_else(|| "/".into()),
        env: req.env,
    }))
}

pub async fn list_sessions(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
) -> ApiResult<ListSessionsResponse> {
    let vm = s
        .store
        .get(&vm_id)
        .await
        .ok_or_else(|| api_err(StatusCode::NOT_FOUND, "VM not found"))?;
    let sessions = vm
        .sessions
        .values()
        .map(|sess| SessionObject {
            session_id: sess.session_id.clone(),
            session_idx: sess.session_idx,
            state: match sess.state {
                SessionState::Idle => "idle",
                SessionState::Running => "running",
            }
            .into(),
            cwd: sess.cwd.clone(),
            env: if sess.env.is_empty() { None } else { Some(sess.env.clone()) },
        })
        .collect();
    Ok(Json(ListSessionsResponse { sessions, next_cursor: None }))
}

pub async fn delete_session(
    State(s): State<AppState>,
    Path((vm_id, session_id)): Path<(String, String)>,
) -> ApiResult<DeleteResponse> {
    let removed = {
        let mut found = false;
        s.store
            .update(&vm_id, |r| {
                found = r.sessions.remove(&session_id).is_some();
            })
            .await;
        found
    };
    if !removed {
        return Err(api_err(StatusCode::NOT_FOUND, "session not found"));
    }
    Ok(Json(DeleteResponse { deleted: true }))
}

// ── Sync ─────────────────────────────────────────────────────────────────────

pub async fn post_sync(
    State(s): State<AppState>,
    Path(vm_id): Path<String>,
    Json(req): Json<SyncRequest>,
) -> ApiResult<SyncResponse> {
    handle_sync(&s, &vm_id, req).await.map(Json).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("VM not found") {
            not_found(e)
        } else if msg.contains("not running") {
            api_err(StatusCode::CONFLICT, e)
        } else {
            internal(e)
        }
    })
}

pub async fn handle_sync(
    s: &AppState,
    vm_id: &str,
    req: SyncRequest,
) -> anyhow::Result<SyncResponse> {
    let vm = s
        .store
        .get(vm_id)
        .await
        .ok_or_else(|| anyhow::anyhow!("VM not found: {vm_id}"))?;
    let Some(handle) = vm.handle else {
        anyhow::bail!("VM is not running; resume it first");
    };

    match req {
        SyncRequest::Read { path } => {
            let out = s
                .backend
                .run(
                    &handle,
                    &format!("cat {}", shell_escape(&path)),
                    "sync_read",
                    &HashMap::new(),
                    "/",
                    Duration::from_secs(30),
                )
                .await?;
            let size = out.stdout.len() as u64;
            let (content, encoding) = encode_output(&out.stdout);
            Ok(SyncResponse::Read {
                ok: true,
                path,
                size,
                content: Some(content),
                encoding: Some(encoding.to_owned()),
                presigned_url: None,
                expires_in: None,
                method: None,
            })
        }
        SyncRequest::Write { writes } => {
            let mut results = Vec::with_capacity(writes.len());
            for w in writes {
                results.push(handle_sync_write(&s.backend, &handle, w).await);
            }
            Ok(SyncResponse::Write { ok: true, results })
        }
    }
}

async fn handle_sync_write(
    backend: &Arc<dyn VmBackend>,
    handle: &crate::backend::VmHandle,
    w: SyncWrite,
) -> SyncWriteResult {
    if w.presigned == Some(true) {
        return SyncWriteResult {
            received_bytes: 0,
            complete: false,
            written: false,
            presigned_url: None,
            upload_id: None,
            expires_in: None,
            method: None,
            error: Some("presigned uploads not supported on this worker".into()),
        };
    }
    let Some(content) = w.content else {
        return SyncWriteResult {
            received_bytes: 0,
            complete: false,
            written: false,
            presigned_url: None,
            upload_id: None,
            expires_in: None,
            method: None,
            error: Some("missing content".into()),
        };
    };
    let bytes = B64.decode(&content).unwrap_or_else(|_| content.into_bytes());
    let b64 = B64.encode(&bytes);
    let cmd = format!(
        "mkdir -p {} && echo '{}' | base64 -d > {}",
        shell_escape(
            std::path::Path::new(&w.path)
                .parent()
                .and_then(|p| p.to_str())
                .unwrap_or("/")
        ),
        b64,
        shell_escape(&w.path)
    );
    let out = backend
        .run(
            handle,
            &cmd,
            "sync_write",
            &HashMap::new(),
            "/",
            Duration::from_secs(30),
        )
        .await;
    let written = out.as_ref().map(|o| o.exit_code == 0).unwrap_or(false);
    SyncWriteResult {
        received_bytes: bytes.len() as u64,
        complete: written,
        written,
        presigned_url: None,
        upload_id: None,
        expires_in: None,
        method: None,
        error: out.err().map(|e| e.to_string()),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn encode_output(bytes: &[u8]) -> (String, &'static str) {
    std::str::from_utf8(bytes).map_or_else(|_| (B64.encode(bytes), "base64"), |s| (s.to_owned(), "utf-8"))
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}


// ── Live / health-check ───────────────────────────────────────────────────────

pub async fn get_live() -> axum::http::StatusCode {
    axum::http::StatusCode::OK
}
