pub mod bootstrap;
pub mod docker;
pub mod hypeman;
mod platform;
pub mod smolvm;

pub use platform::worker_platform;

use std::{collections::HashMap, path::Path, process::Stdio, time::Duration};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncReadExt, process::Command};
use tokio_util::sync::CancellationToken;

pub use crate::api::types::RegistryAuth;

#[derive(Debug, Clone)]
pub struct VmHandle {
    pub id: String,
}

/// Extract the registry host from an OCI image reference, following the same
/// convention Docker and other OCI tooling use: the first path segment before
/// a `/` is a registry host only when it looks like one (contains `.` or `:`,
/// or is exactly `localhost`); otherwise the reference is an (optionally
/// namespaced) Docker Hub image, and `"docker.io"` is returned.
///
/// Examples: `"ghcr.io/org/image"` -> `"ghcr.io"`; `"ubuntu:24.04"` ->
/// `"docker.io"`; `"myuser/private-repo"` -> `"docker.io"`;
/// `"registry.example.com:5000/image"` -> `"registry.example.com:5000"`.
pub fn extract_registry(image: &str) -> &str {
    if let Some(slash) = image.find('/') {
        let candidate = &image[..slash];
        if candidate.contains('.') || candidate.contains(':') || candidate == "localhost" {
            return candidate;
        }
    }
    "docker.io"
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resources {
    pub vcpu: u32,
    pub memory_mib: u64,
    pub disk_mib: u64,
}

/// One command to run in a VM.
pub struct Exec<'a> {
    pub command: &'a str,
    /// Added to the image's environment.
    pub env: &'a HashMap<String, String>,
    /// `None` starts the command in the image's working directory.
    pub cwd: Option<&'a str>,
    /// `None` lets the command run without a limit.
    pub timeout: Option<Duration>,
    pub stdout: Stdout<'a>,
    /// Ends the command early; it then counts as cancelled.
    pub cancel: &'a CancellationToken,
}

/// Where a command's standard output goes.
pub enum Stdout<'a> {
    /// Kept in memory up to [`OUTPUT_LIMIT`], like standard error.
    Capture,
    /// Written whole to this host file.
    File(&'a Path),
}

/// How long one of BLAST's own guest commands (a file read or write) may take.
const INTERNAL_EXEC_TIMEOUT: Duration = Duration::from_secs(60);

static NO_ENV: std::sync::LazyLock<HashMap<String, String>> =
    std::sync::LazyLock::new(HashMap::new);
static NEVER_CANCELLED: std::sync::LazyLock<CancellationToken> =
    std::sync::LazyLock::new(CancellationToken::new);

impl<'a> Exec<'a> {
    /// One of BLAST's own guest commands, run from `/`.
    pub fn internal(command: &'a str, stdout: Stdout<'a>) -> Self {
        Self {
            command,
            env: &NO_ENV,
            cwd: Some("/"),
            timeout: Some(INTERNAL_EXEC_TIMEOUT),
            stdout,
            cancel: &NEVER_CANCELLED,
        }
    }
}

/// Bytes of each output stream a run keeps: what one run record carries.
pub const OUTPUT_LIMIT: usize = 64 * 1024;

/// The start of an output stream, and whether there was more.
#[derive(Debug, Default)]
pub struct Captured {
    pub bytes: Vec<u8>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ended {
    Exited(i32),
    /// Killed at its timeout.
    TimedOut,
    Cancelled,
}

impl Ended {
    /// A command killed at its timeout reports `timeout(1)`'s exit code; a
    /// cancelled one has none.
    pub const fn exit_code(self) -> Option<i32> {
        match self {
            Self::Exited(code) => Some(code),
            Self::TimedOut => Some(124),
            Self::Cancelled => None,
        }
    }
}

/// What a command printed before it ended, however it ended.
#[derive(Debug)]
pub struct RunOutput {
    pub stdout: Captured,
    pub stderr: Captured,
    pub ended: Ended,
}

#[cfg(test)]
impl RunOutput {
    /// A command cancelled before it printed anything.
    pub fn cancelled() -> Self {
        Self {
            stdout: Captured::default(),
            stderr: Captured::default(),
            ended: Ended::Cancelled,
        }
    }
}

/// Runs `command` (a backend's exec client) for `exec`, collecting what it
/// prints as it prints it, so a command stopped at its timeout or cancelled
/// keeps its output so far. Stopping kills the client, which need not end
/// the guest command: [`run_supervised`] does.
pub async fn exec_client(mut command: Command, exec: &Exec<'_>) -> Result<RunOutput> {
    let stdout = match exec.stdout {
        Stdout::Capture => Stdio::piped(),
        Stdout::File(path) => std::fs::File::create(path)
            .with_context(|| format!("create {}", path.display()))?
            .into(),
    };
    // Its own process group, so stopping it stops whatever it started too
    // and the output pipes close.
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("start exec client")?;
    let (stdout, stderr) = (child.stdout.take(), child.stderr.take());
    let output = async { tokio::join!(capture(stdout), capture(stderr)) };
    tokio::pin!(output);
    let deadline = async {
        match exec.timeout {
            Some(limit) => tokio::time::sleep(limit).await,
            None => std::future::pending().await,
        }
    };
    let finished = tokio::select! {
        // Both pipes close when the client exits.
        captured = &mut output => Ok(captured),
        () = deadline => Err(Ended::TimedOut),
        () = exec.cancel.cancelled() => Err(Ended::Cancelled),
    };
    let (stopped, (stdout, stderr)) = match finished {
        Ok(captured) => (None, captured),
        Err(stopped) => {
            kill_group(&mut child).context("stop exec client")?;
            (Some(stopped), output.await)
        }
    };
    let status = child.wait().await.context("wait for exec client")?;
    Ok(RunOutput {
        stdout: stdout?,
        stderr: stderr?,
        ended: stopped.unwrap_or_else(|| Ended::Exited(status.code().unwrap_or(-1))),
    })
}

fn kill_group(child: &mut tokio::process::Child) -> std::io::Result<()> {
    #[cfg(unix)]
    if let Some(group) = child.id().and_then(|pid| libc::pid_t::try_from(pid).ok()) {
        // SAFETY: `killpg` only sends a signal; the group is the child's own.
        if unsafe { libc::killpg(group, libc::SIGKILL) } == 0 {
            return Ok(());
        }
    }
    child.start_kill()
}

/// Reads a stream to its end, keeping its first [`OUTPUT_LIMIT`] bytes.
async fn capture(stream: Option<impl tokio::io::AsyncRead + Unpin>) -> Result<Captured> {
    let mut captured = Captured::default();
    let Some(mut stream) = stream else {
        return Ok(captured);
    };
    let mut buf = vec![0u8; 16 * 1024];
    loop {
        let n = stream.read(&mut buf).await.context("read exec output")?;
        if n == 0 {
            return Ok(captured);
        }
        let kept = n.min(OUTPUT_LIMIT - captured.bytes.len());
        captured.bytes.extend_from_slice(&buf[..kept]);
        captured.truncated |= kept < n;
    }
}

#[async_trait]
pub trait VmBackend: Send + Sync + 'static {
    /// Boot a fresh VM from an OCI image reference. `registry_auth`, when
    /// present, authenticates the pull against a private registry.
    async fn fork_image(
        &self,
        image: &str,
        resources: &Resources,
        registry_auth: Option<&RegistryAuth>,
    ) -> Result<VmHandle>;

    /// Restore a VM from a snapshot directory.
    async fn fork_snapshot(&self, snap_dir: &Path, resources: &Resources) -> Result<VmHandle>;

    /// Run a command inside a running VM, in the image's environment.
    async fn run(&self, vm: &VmHandle, exec: Exec<'_>) -> Result<RunOutput>;

    /// Freeze the VM: stop CPU execution while keeping memory resident.
    /// Called only when [`VmBackend::supports_pause`] is true.
    async fn pause(&self, vm: &VmHandle) -> Result<()>;

    /// Resume a paused VM (CPU unfrozen; memory already resident).
    async fn unpause(&self, vm: &VmHandle) -> Result<()>;

    /// Whether `pause` really freezes CPU while retaining a live VM. A
    /// backend without that primitive is suspended directly; it must never
    /// be accounted as Paused while its CPU is still running.
    fn supports_pause(&self) -> bool {
        false
    }

    /// Host memory each live VM holds beyond its guest RAM, which pool
    /// admission must charge or the host runs out before the pool does.
    fn vm_memory_overhead_mib(&self) -> u64 {
        0
    }

    /// Write VM state to `snap_dir`, leaving the VM running (dirty-sync snapshot).
    /// Files already in `snap_dir` are replaced, never rewritten in place: a
    /// fork may still be reading them through hard links.
    async fn snapshot(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()>;

    /// Removes what the backend still holds of machines that are stopped and
    /// that no VM in `kept` runs on, such as a fork that failed halfway.
    /// Returns how many it removed.
    async fn remove_stopped_leftovers(
        &self,
        _kept: &std::collections::HashSet<String>,
    ) -> Result<usize> {
        Ok(0)
    }

    /// [`Self::snapshot`], sending on `resumed` as soon as the VM runs again:
    /// a backend may pause it for a moment and go on writing the snapshot for
    /// much longer. One that cannot tell the two apart sends when it is done.
    async fn snapshot_while_running(
        &self,
        vm: &VmHandle,
        snap_dir: &Path,
        resumed: tokio::sync::oneshot::Sender<()>,
    ) -> Result<()> {
        let written = self.snapshot(vm, snap_dir).await;
        let _ = resumed.send(());
        written
    }

    /// Suspend: write state to `snap_dir` as `snapshot` does, and destroy the
    /// VM process.
    async fn suspend(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()>;

    /// Restore a suspended VM from `snap_dir`.
    async fn resume(&self, snap_dir: &Path, resources: &Resources) -> Result<VmHandle>;

    /// Destroy the VM and release all resources.
    async fn delete(&self, vm: &VmHandle) -> Result<()>;

    /// Whether a backend handle recorded before a worker restart still names
    /// a running VM on this machine. Backends whose handles are process-local
    /// keep the conservative default; daemon-backed hypervisors can reattach.
    async fn can_recover(&self, _vm: &VmHandle) -> Result<bool> {
        Ok(false)
    }

    /// Whether a VM's machine is known to have stopped, so it cannot be
    /// checkpointed and only its last snapshot can bring it back. Backends
    /// that cannot tell say no.
    async fn is_stopped(&self, _vm: &VmHandle) -> Result<bool> {
        Ok(false)
    }

    /// Write the host file `src` to `guest_path` inside a running VM,
    /// creating parent directories and replacing the destination only once
    /// every byte has landed. The default goes through `run` in bounded
    /// chunks, so it works on any backend whatever the file size; backends
    /// that can stream stdin into an exec override it. A destination that
    /// is a directory fails with [`IS_A_DIRECTORY`] in the error.
    async fn write_file(
        &self,
        vm: &VmHandle,
        guest_path: &str,
        src: &Path,
        mode: u32,
    ) -> Result<()> {
        let mut file = tokio::fs::File::open(src).await?;
        write_in_chunks(self, vm, guest_path, &mut file, mode).await
    }

    /// e.g. "docker", "smolvm", "hypeman". This
    /// identifies the snapshot FORMAT: a Docker commit-based image, a
    /// `SmolVM` memory snapshot, and a Hypeman snapshot are mutually
    /// unreadable, so a snapshot written by one backend can never be
    /// resumed by another. See `SnapshotStore::{write,check}_backend_tag`.
    fn kind(&self) -> &'static str;

    /// Whether a snapshot directory holds the whole VM, so it can be stored
    /// and resumed by another worker. A backend whose snapshots point at
    /// state kept elsewhere says no, and its snapshots stay local.
    fn snapshots_are_portable(&self) -> bool {
        true
    }
}

/// Runs a caller's command under a supervisor in the guest, so that it runs
/// as the API's commands do whatever the backend's exec does:
///
/// - in bash where the image has it, and in `sh` where it does not;
/// - with no input to read;
/// - as the leader of its own process group, which a timeout or a cancel
///   kills whole, where the backend alone would leave it running;
/// - printing to files in the guest, of which the run keeps the start, so a
///   command may print more than the backend's exec can carry.
///
/// `id` names the run among those of its VM.
pub async fn run_supervised<B: VmBackend + ?Sized>(
    backend: &B,
    vm: &VmHandle,
    exec: Exec<'_>,
    id: &str,
) -> Result<RunOutput> {
    // One argument of an exec is bounded by the kernel, so a long command
    // goes into the guest as a file its shell reads.
    let staged;
    let command = if exec.command.len() > MAX_ARGUMENT_COMMAND {
        let script = format!("/tmp/.blast-run-{id}.cmd");
        write_in_chunks(backend, vm, &script, &mut exec.command.as_bytes(), 0o600).await?;
        staged = format!(". {}", shell_quote(&script));
        staged.as_str()
    } else {
        exec.command
    };
    let supervised = supervise_command(command, id);
    let out = backend
        .run(
            vm,
            Exec {
                command: &supervised,
                ..exec
            },
        )
        .await?;
    if matches!(out.ended, Ended::Exited(_)) {
        return Ok(out);
    }
    // The backend stopped waiting; the command is still running in the guest.
    let stopped = backend
        .run(vm, Exec::internal(&stop_command(id), Stdout::Capture))
        .await?;
    Ok(RunOutput {
        ended: out.ended,
        ..stopped
    })
}

/// The longest command passed as an exec argument: under the kernel's 128 KiB
/// for one argument, with room for its quoting.
const MAX_ARGUMENT_COMMAND: usize = 64 * 1024;

/// Where run `id` keeps its process id and what it printed.
fn run_dir(id: &str) -> String {
    shell_quote(&format!("/tmp/.blast-run-{id}"))
}

/// One more byte than a run keeps tells a stream at the limit from a longer one.
const KEPT: usize = OUTPUT_LIMIT + 1;

fn supervise_command(command: &str, id: &str) -> String {
    format!(
        "sh -c {} sh {} {} {KEPT}",
        shell_quote(SUPERVISE),
        shell_quote(command),
        run_dir(id)
    )
}

fn stop_command(id: &str) -> String {
    format!("sh -c {} sh {} {KEPT}", shell_quote(STOP), run_dir(id))
}

/// `$1` the command, `$2` the run's directory, `$3` the bytes of each stream
/// to print. Without the directory the command runs unsupervised.
const SUPERVISE: &str = r#"shell=sh; command -v bash >/dev/null 2>&1 && shell=bash
mkdir -p "$2" 2>/dev/null || exec "$shell" -c "$1"
if command -v setsid >/dev/null 2>&1; then
    ( exec setsid "$shell" -c "$1" ) </dev/null >"$2/out" 2>"$2/err" &
else
    ( exec "$shell" -c "$1" ) </dev/null >"$2/out" 2>"$2/err" &
fi
echo $! >"$2/pid"
wait $!
status=$?
head -c "$3" "$2/out" 2>/dev/null
head -c "$3" "$2/err" >&2 2>/dev/null
rm -rf "$2" "$2.cmd"
exit "$status""#;

/// `$1` the run's directory, `$2` the bytes of each stream to print. Takes
/// the directory from the supervisor first, so what the command printed is
/// still there to read once it is killed.
const STOP: &str = r#"mv "$1" "$1.stopped" 2>/dev/null || exit 0
pid=$(cat "$1.stopped/pid" 2>/dev/null)
if [ -n "$pid" ]; then
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
fi
head -c "$2" "$1.stopped/out" 2>/dev/null
head -c "$2" "$1.stopped/err" >&2 2>/dev/null
rm -rf "$1.stopped" "$1.cmd""#;

/// Quotes `s` as one POSIX shell word.
pub fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Raw bytes per `run` call: base64 inflates this to 64 KiB, well under the
/// kernel's 128 KiB limit on a single exec argument.
const WRITE_CHUNK_BYTES: usize = 48 * 1024;

/// What a file write reports when its destination is a directory.
pub const IS_A_DIRECTORY: &str = "destination is a directory";

async fn write_in_chunks<B: VmBackend + ?Sized>(
    backend: &B,
    vm: &VmHandle,
    guest_path: &str,
    file: &mut (impl tokio::io::AsyncRead + Unpin + Send),
    mode: u32,
) -> Result<()> {
    let partial = shell_quote(&format!("{guest_path}.blast-partial"));
    let dest = shell_quote(guest_path);
    let sh = |command: String| async move {
        let out = backend
            .run(vm, Exec::internal(&command, Stdout::Capture))
            .await?;
        if out.ended != Ended::Exited(0) {
            bail!(
                "write {guest_path}: {:?}: {}",
                out.ended,
                String::from_utf8_lossy(&out.stderr.bytes).trim()
            );
        }
        Ok(())
    };
    let written = async {
        sh(format!(
            "if [ -d {dest} ]; then echo {} >&2; exit 21; fi; \
             mkdir -p \"$(dirname {dest})\" && : > {partial}",
            shell_quote(IS_A_DIRECTORY)
        ))
        .await?;
        let mut buf = vec![0u8; WRITE_CHUNK_BYTES];
        loop {
            let mut filled = 0;
            while filled < buf.len() {
                let n = file.read(&mut buf[filled..]).await?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            if filled == 0 {
                break;
            }
            let b64 = B64.encode(&buf[..filled]);
            sh(format!("printf %s {b64} | base64 -d >> {partial}")).await?;
            if filled < buf.len() {
                break;
            }
        }
        sh(format!(
            "chmod {mode:o} {partial} && mv -f {partial} {dest}"
        ))
        .await
    }
    .await;
    if written.is_err() {
        // A file that did not land leaves nothing beside its destination.
        let _ = sh(format!("rm -f {partial}")).await;
    }
    written
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn sh(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    fn exec<'a>(
        timeout: Option<Duration>,
        stdout: Stdout<'a>,
        cancel: &'a CancellationToken,
    ) -> Exec<'a> {
        Exec {
            command: "",
            env: &NO_ENV,
            cwd: None,
            timeout,
            stdout,
            cancel,
        }
    }

    #[tokio::test]
    async fn a_command_killed_at_its_timeout_keeps_what_it_printed() {
        let cancel = CancellationToken::new();
        let out = exec_client(
            sh("echo STARTED; echo WARNED >&2; sleep 30"),
            &exec(Some(Duration::from_millis(500)), Stdout::Capture, &cancel),
        )
        .await
        .unwrap();

        assert_eq!(out.ended, Ended::TimedOut);
        assert_eq!(out.ended.exit_code(), Some(124));
        assert_eq!(out.stdout.bytes, b"STARTED\n");
        assert_eq!(out.stderr.bytes, b"WARNED\n");
    }

    #[tokio::test]
    async fn a_cancelled_command_keeps_what_it_printed_and_has_no_exit_code() {
        let cancel = CancellationToken::new();
        let stop = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            stop.cancel();
        });
        let out = exec_client(
            sh("echo STARTED; sleep 30"),
            &exec(None, Stdout::Capture, &cancel),
        )
        .await
        .unwrap();

        assert_eq!(out.ended, Ended::Cancelled);
        assert_eq!(out.ended.exit_code(), None);
        assert_eq!(out.stdout.bytes, b"STARTED\n");
    }

    #[tokio::test]
    async fn output_past_the_limit_is_dropped_and_marked() {
        let cancel = CancellationToken::new();
        let out = exec_client(
            sh("head -c 200000 /dev/zero; echo short >&2; exit 3"),
            &exec(None, Stdout::Capture, &cancel),
        )
        .await
        .unwrap();

        assert_eq!(out.ended, Ended::Exited(3));
        assert_eq!(out.stdout.bytes.len(), OUTPUT_LIMIT);
        assert!(out.stdout.truncated);
        assert_eq!(out.stderr.bytes, b"short\n");
        assert!(!out.stderr.truncated);
    }

    #[tokio::test]
    async fn output_sent_to_a_file_is_whole() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("out");
        let cancel = CancellationToken::new();
        let out = exec_client(
            sh("head -c 200000 /dev/zero"),
            &exec(None, Stdout::File(&file), &cancel),
        )
        .await
        .unwrap();

        assert_eq!(out.ended, Ended::Exited(0));
        assert_eq!(std::fs::metadata(&file).unwrap().len(), 200_000);
    }
    #[tokio::test]
    async fn a_supervised_command_stopped_at_its_timeout_is_killed_whole_and_keeps_what_it_printed()
    {
        let id = ulid::Ulid::new().to_string();
        let tag = format!("7{}", std::process::id());
        let cancel = CancellationToken::new();
        let command = format!("echo STARTED; sleep {tag}1 & sleep {tag}2");

        let out = exec_client(
            sh(&supervise_command(&command, &id)),
            &exec(Some(Duration::from_secs(1)), Stdout::Capture, &cancel),
        )
        .await
        .unwrap();
        assert_eq!(out.ended, Ended::TimedOut);
        let stopped = exec_client(
            sh(&stop_command(&id)),
            &exec(None, Stdout::Capture, &cancel),
        )
        .await
        .unwrap();

        assert_eq!(stopped.stdout.bytes, b"STARTED\n");
        let left = exec_client(
            sh(&format!("pgrep -f 'sleep {tag}[12]' | wc -l")),
            &exec(None, Stdout::Capture, &cancel),
        )
        .await
        .unwrap();
        assert_eq!(String::from_utf8_lossy(&left.stdout.bytes).trim(), "0");
    }

    #[tokio::test]
    async fn a_supervised_command_prints_past_the_limit_and_still_ends_as_it_exits() {
        let id = ulid::Ulid::new().to_string();
        let cancel = CancellationToken::new();
        let command = format!(
            "head -c {} /dev/zero; echo late >&2; exit 3",
            4 * OUTPUT_LIMIT
        );

        let out = exec_client(
            sh(&supervise_command(&command, &id)),
            &exec(None, Stdout::Capture, &cancel),
        )
        .await
        .unwrap();

        assert_eq!(out.ended, Ended::Exited(3));
        assert_eq!(
            (out.stdout.bytes.len(), out.stdout.truncated),
            (OUTPUT_LIMIT, true)
        );
        assert_eq!(
            (out.stderr.bytes.as_slice(), out.stderr.truncated),
            (&b"late\n"[..], false)
        );
    }
}
