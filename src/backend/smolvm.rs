use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use tokio::{process::Command, sync::Semaphore};

use super::{Captured, Ended, Exec, RegistryAuth, Resources, RunOutput, VmBackend, VmHandle};

/// Fields returned by `smolvm machine ls --json`.
#[derive(serde::Deserialize)]
struct MachineInfo {
    name: String,
    cpus: u32,
    memory_mib: u64,
    /// Absent from releases that did not report it.
    #[serde(default)]
    state: Option<String>,
}

impl MachineInfo {
    fn is_running(&self) -> bool {
        self.state
            .as_deref()
            .map_or(true, |state| state == "running")
    }
}

pub struct SmolvmBackend {
    binary: std::path::PathBuf,
    /// Root for `SmolVM` images, checkpoints, and cache files.
    smolvm_data_dir: std::path::PathBuf,
    /// Each machine's share of its guest agent's connections; see [`EXEC_SLOTS`].
    exec_slots: Mutex<HashMap<String, Arc<Semaphore>>>,
}

/// `SmolVM`'s guest agent serves this many connections at once (its
/// `MAX_CONCURRENT_CONNECTIONS`) and accepts none while they are all in use:
/// one command too many, and the machine looks stopped to every command,
/// `SmolVM`'s own state probe among them.
const AGENT_CONNECTIONS: usize = 8;

/// How many of a machine's agent connections commands run on. The rest stay
/// free for what else is sent to it while they run: state probes and
/// checkpoints.
const EXEC_SLOTS: usize = AGENT_CONNECTIONS - 2;

/// How long a `SmolVM` command waits out a locked state database.
const DATABASE_LOCKED_RETRY: Duration = Duration::from_secs(30);

/// Between half and all of `backoff`, so contending commands spread out.
fn jittered(backoff: Duration) -> Duration {
    let half = backoff / 2;
    let spread = u64::try_from(half.as_micros()).unwrap_or(u64::MAX).max(1);
    let random = u64::try_from(ulid::Ulid::new().random() % u128::from(spread)).unwrap_or(0);
    half + Duration::from_micros(random)
}

/// What a failed `SmolVM` command says went wrong: its `Error:` line, without
/// the progress it printed on the way there, which names its machines.
fn failure(stderr: &str) -> &str {
    let lines = || {
        stderr
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
    };
    lines()
        .rev()
        .find_map(|line| line.strip_prefix("Error: "))
        .or_else(|| lines().next_back())
        .unwrap_or_default()
}

fn flag_value<'a>(args: &[&'a str], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1])
}

/// Escape a string for embedding in a TOML basic string (`"..."`).
fn toml_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

impl SmolvmBackend {
    pub fn new(binary: std::path::PathBuf, data_dir: std::path::PathBuf) -> Self {
        Self {
            binary,
            smolvm_data_dir: data_dir,
            exec_slots: Mutex::default(),
        }
    }

    fn exec_slots(&self, machine: &str) -> Arc<Semaphore> {
        self.exec_slots
            .lock()
            .expect("exec slots lock")
            .entry(machine.to_owned())
            .or_insert_with(|| Arc::new(Semaphore::new(EXEC_SLOTS)))
            .clone()
    }

    /// A machine restored from a checkpoint wakes with the clock the
    /// checkpoint was taken at, and on Linux nothing moves it on: each
    /// suspension would leave it that much further behind. A guest whose
    /// clock cannot be set keeps running on the old one.
    async fn set_clock(&self, machine: &str) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_secs());
        let set = format!("date -u -s @{now} >/dev/null");
        let outcome = self
            .smolvm(&["machine", "exec", "--name", machine, "--", "sh", "-c", &set])
            .await;
        match outcome {
            Ok(out) if out.status.success() => {}
            Ok(out) => tracing::warn!(
                machine,
                err = %String::from_utf8_lossy(&out.stderr).trim(),
                "setting a restored machine's clock failed"
            ),
            Err(error) => {
                tracing::warn!(machine, err = %format!("{error:#}"), "setting a restored machine's clock failed");
            }
        }
    }

    async fn smolvm(&self, args: &[&str]) -> Result<std::process::Output> {
        self.smolvm_env(args, None).await
    }

    /// Runs a `SmolVM` command, retrying while its state database is locked:
    /// concurrent commands contend for it and give up quickly. A command
    /// that fails this way has changed nothing (a failed `create` removes
    /// what it made), so it can simply run again.
    async fn smolvm_env(
        &self,
        args: &[&str],
        config_path: Option<&Path>,
    ) -> Result<std::process::Output> {
        let deadline = std::time::Instant::now() + DATABASE_LOCKED_RETRY;
        let mut backoff = Duration::from_millis(50);
        loop {
            let out = self.smolvm_once(args, config_path).await?;
            if out.status.success()
                || !String::from_utf8_lossy(&out.stderr).contains("database is locked")
                || std::time::Instant::now() >= deadline
            {
                return Ok(out);
            }
            if let Some(output) = flag_value(args, "--output") {
                // A checkpoint that got as far as its output would refuse
                // to overwrite it.
                let _ = std::fs::remove_file(output);
            }
            tokio::time::sleep(jittered(backoff)).await;
            backoff = (backoff * 2).min(Duration::from_secs(2));
        }
    }

    async fn smolvm_once(
        &self,
        args: &[&str],
        config_path: Option<&Path>,
    ) -> Result<std::process::Output> {
        let mut cmd = self.command();
        if let Some(p) = config_path {
            cmd.env("SMOLVM_CONFIG", p);
        }
        cmd.args(args).output().await.map_err(Into::into)
    }

    /// The `SmolVM` binary, pointed at this worker's data directory.
    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.binary);
        cmd.env("XDG_DATA_HOME", &self.smolvm_data_dir);
        cmd.env(
            "SMOLVM_PACK_STAGING",
            self.smolvm_data_dir.join("pack-staging"),
        );
        cmd.env("XDG_CACHE_HOME", self.smolvm_data_dir.join("cache"));
        cmd
    }

    fn check(out: std::process::Output, ctx: &str) -> Result<std::process::Output> {
        if out.status.success() {
            Ok(out)
        } else {
            bail!("{ctx}: {}", failure(&String::from_utf8_lossy(&out.stderr)))
        }
    }

    /// Writes registry credentials scoped to one image pull.
    async fn write_scoped_registry_config(image: &str, auth: &RegistryAuth) -> Result<PathBuf> {
        let registry = super::extract_registry(image);
        let dir = std::env::temp_dir().join(format!("blast-smolvm-auth-{}", ulid::Ulid::new()));
        tokio::fs::create_dir_all(&dir).await?;
        let config_path = dir.join("config.toml");
        let contents = format!(
            "[images.registries.\"{}\"]\nusername = \"{}\"\npassword = \"{}\"\n",
            toml_escape(registry),
            toml_escape(&auth.username),
            toml_escape(&auth.password),
        );
        tokio::fs::write(&config_path, contents).await?;
        Ok(config_path)
    }

    /// Reads back a just-created machine's actual cpus/memory from
    /// `machine ls --json` -- the only way to know what a checkpoint
    /// restore actually produced, since `create --from` takes no sizing
    /// flags of its own to echo back.
    async fn machine_info(&self, name: &str) -> Result<MachineInfo> {
        self.listed(name)
            .await?
            .ok_or_else(|| anyhow::anyhow!("smolvm machine ls: {name} not found after create"))
    }

    async fn listed(&self, name: &str) -> Result<Option<MachineInfo>> {
        let out = Self::check(
            self.smolvm(&["machine", "ls", "--json"]).await?,
            "smolvm machine ls",
        )?;
        let list: Vec<MachineInfo> =
            serde_json::from_slice(&out.stdout).context("parse smolvm machine ls --json")?;
        Ok(list.into_iter().find(|m| m.name == name))
    }
}

#[async_trait]
impl VmBackend for SmolvmBackend {
    async fn fork_image(
        &self,
        image: &str,
        resources: &Resources,
        registry_auth: Option<&RegistryAuth>,
    ) -> Result<VmHandle> {
        let id = ulid::Ulid::new().to_string();
        let mem = resources.memory_mib.to_string();
        let storage_gib = resources.disk_mib.div_ceil(1024).max(1).to_string();

        let cred_file = match registry_auth {
            Some(auth) => Some(Self::write_scoped_registry_config(image, auth).await?),
            None => None,
        };
        // `machine create` only registers the config; for an `--image`
        // machine the actual registry pull happens lazily on the first
        // `machine start`, so the scoped credential has to be in scope for
        // both calls, not just `create`.
        let run_result: Result<()> = async {
            Self::check(
                self.smolvm_env(
                    &[
                        "machine",
                        "create",
                        "--name",
                        &id,
                        "--image",
                        image,
                        "--net",
                        "--cpus",
                        &resources.vcpu.to_string(),
                        "--mem",
                        &mem,
                        "--storage",
                        &storage_gib,
                    ],
                    cred_file.as_deref(),
                )
                .await?,
                "smolvm create",
            )?;
            Self::check(
                self.smolvm_env(&["machine", "start", "--name", &id], cred_file.as_deref())
                    .await?,
                "smolvm start",
            )?;
            Ok(())
        }
        .await;

        if let Some(path) = &cred_file {
            if let Some(dir) = path.parent() {
                // Never leave the scoped credential behind longer than the
                // create+start pair that needed it.
                let _ = tokio::fs::remove_dir_all(dir).await;
            }
        }
        if let Err(error) = run_result {
            // `create` registered the machine before `start` failed.
            let _ = self
                .smolvm(&["machine", "delete", "--name", &id, "--force"])
                .await;
            return Err(error);
        }
        Ok(VmHandle { id })
    }

    async fn fork_snapshot(&self, snap_dir: &Path, resources: &Resources) -> Result<VmHandle> {
        let id = ulid::Ulid::new().to_string();
        let checkpoint_path = snap_dir.join("machine.smolcheckpoint");
        // A live checkpoint captures CPU/memory/disk/device topology as one
        // unit. `create --from` refuses ANY of --cpus/--mem/--storage/--net
        // on top of it outright, and `machine update` afterward is a false
        // hope: the CLI accepts the metadata change (exit 0, "cpus: 1 -> 2"),
        // but starting that machine then fails hard --
        // `krun_start_enter returned: -22 (EINVAL)` -- because the
        // checkpoint's own memory-mapped device layout was baked at its
        // original size and libkrun cannot boot a mismatch (confirmed
        // directly: even a cpu-only change with memory/disk untouched still
        // fails to start). A VM-fork on this backend is always exactly the
        // size of its source; there is no path to deliver anything else.
        Self::check(
            self.smolvm(&[
                "machine",
                "create",
                "--name",
                &id,
                "--from",
                &checkpoint_path.to_string_lossy(),
            ])
            .await?,
            "smolvm create --from",
        )?;
        // Confirm the caller's request actually matches what was inherited
        // before starting -- handlers.rs already resolves an omitted
        // `resources` to the source's own shape, so this only trips on a
        // caller that explicitly asked for something different, and fails
        // clearly here rather than with a cryptic EINVAL at boot.
        let restored = self.machine_info(&id).await?;
        let (restored_vcpu, restored_memory_mib) = (restored.cpus, restored.memory_mib);
        if restored_vcpu != resources.vcpu || restored_memory_mib != resources.memory_mib {
            let _ = self
                .smolvm(&["machine", "delete", "--name", &id, "--force"])
                .await;
            bail!(
                "cannot resize on fork from a running VM: source is {}vcpu/{}MiB, requested {}vcpu/{}MiB. SmolVM's live-checkpoint restore always inherits the source's exact topology; omit `resources` to inherit it, or fork from an image instead if a specific size is required.",
                restored.cpus, restored.memory_mib, resources.vcpu, resources.memory_mib
            );
        }
        Self::check(
            self.smolvm(&["machine", "start", "--name", &id]).await?,
            "smolvm start",
        )?;
        self.set_clock(&id).await;
        Ok(VmHandle { id })
    }

    /// `--stream` relays output as the command prints it, with no size
    /// limit. Without `--workdir` the command starts in the image's working
    /// directory; the image's environment applies either way.
    async fn run(&self, vm: &VmHandle, exec: Exec<'_>) -> Result<RunOutput> {
        // A command past the machine's slots waits for one, as long as it
        // may run at all.
        let slots = self.exec_slots(&vm.id);
        let waited_out = |ended| RunOutput {
            stdout: Captured::default(),
            stderr: Captured::default(),
            ended,
        };
        let _slot = tokio::select! {
            slot = slots.acquire() => slot.context("exec slots closed")?,
            () = exec.cancel.cancelled() => return Ok(waited_out(Ended::Cancelled)),
            () = async {
                match exec.timeout {
                    Some(limit) => tokio::time::sleep(limit).await,
                    None => std::future::pending().await,
                }
            } => return Ok(waited_out(Ended::TimedOut)),
        };
        let mut command = self.command();
        command.args(["machine", "exec", "--name", &vm.id, "--stream"]);
        if let Some(cwd) = exec.cwd {
            command.args(["--workdir", cwd]);
        }
        for (key, value) in exec.env {
            command.args(["--env", &format!("{key}={value}")]);
        }
        command.args(["--", "sh", "-c", exec.command]);
        super::exec_client(command, &exec).await
    }

    /// `SmolVM` lets each VMM's cgroup grow to this much past guest RAM
    /// (`VMM_MEM_OVERHEAD_MIB` in its src/process.rs).
    fn vm_memory_overhead_mib(&self) -> u64 {
        768
    }

    // smolvm has no distinct pause; lifecycle calls suspend directly.
    async fn pause(&self, _vm: &VmHandle) -> Result<()> {
        Ok(())
    }

    async fn unpause(&self, _vm: &VmHandle) -> Result<()> {
        Ok(())
    }

    async fn snapshot(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(snap_dir)?;
        // SmolVM refuses to overwrite its output, and a fork may still be
        // reading the previous checkpoint: write aside, then replace it.
        let partial = snap_dir.join("machine.partial.smolcheckpoint");
        match std::fs::remove_file(&partial) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error.into()),
            _ => {}
        }
        Self::check(
            self.smolvm(&[
                "machine",
                "checkpoint",
                "--name",
                &vm.id,
                "--output",
                &partial.to_string_lossy(),
            ])
            .await?,
            "smolvm checkpoint",
        )?;
        std::fs::rename(&partial, snap_dir.join("machine.smolcheckpoint"))?;
        Ok(())
    }

    async fn remove_stopped_leftovers(
        &self,
        kept: &std::collections::HashSet<String>,
    ) -> Result<usize> {
        let out = Self::check(
            self.smolvm(&["machine", "ls", "--json"]).await?,
            "smolvm machine ls",
        )?;
        let machines: Vec<MachineInfo> =
            serde_json::from_slice(&out.stdout).context("parse smolvm machine ls --json")?;
        let mut removed = 0;
        for machine in machines {
            if machine.is_running() || kept.contains(&machine.name) {
                continue;
            }
            Self::check(
                self.smolvm(&["machine", "delete", "--name", &machine.name, "--force"])
                    .await?,
                "smolvm machine delete",
            )?;
            removed += 1;
        }
        Ok(removed)
    }

    /// `SmolVM` saves the machine while it is paused, lets it run again, and
    /// only then packs what it saved into its output, starting with a
    /// temporary file beside it: the first new entry in `snap_dir` says the
    /// machine is running.
    async fn snapshot_while_running(
        &self,
        vm: &VmHandle,
        snap_dir: &Path,
        resumed: tokio::sync::oneshot::Sender<()>,
    ) -> Result<()> {
        std::fs::create_dir_all(snap_dir)?;
        let entries = || -> Result<std::collections::BTreeSet<std::ffi::OsString>> {
            std::fs::read_dir(snap_dir)?
                .map(|entry| Ok(entry?.file_name()))
                .collect()
        };
        match std::fs::remove_file(snap_dir.join("machine.partial.smolcheckpoint")) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error.into()),
            _ => {}
        }
        let before = entries()?;
        let written = self.snapshot(vm, snap_dir);
        tokio::pin!(written);
        let mut resumed = Some(resumed);
        let mut look = tokio::time::interval(Duration::from_millis(50));
        loop {
            tokio::select! {
                written = &mut written => {
                    if let Some(resumed) = resumed.take() {
                        let _ = resumed.send(());
                    }
                    return written;
                }
                _ = look.tick(), if resumed.is_some() => {
                    if entries()? != before {
                        let _ = resumed.take().expect("checked").send(());
                    }
                }
            }
        }
    }

    async fn suspend(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()> {
        self.snapshot(vm, snap_dir).await?;
        self.delete(vm).await
    }

    async fn resume(&self, snap_dir: &Path, resources: &Resources) -> Result<VmHandle> {
        self.fork_snapshot(snap_dir, resources).await
    }

    async fn delete(&self, vm: &VmHandle) -> Result<()> {
        self.exec_slots
            .lock()
            .expect("exec slots lock")
            .remove(&vm.id);
        if self.listed(&vm.id).await?.is_none() {
            return Ok(());
        }
        Self::check(
            self.smolvm(&["machine", "delete", "--name", &vm.id, "--force"])
                .await?,
            "smolvm machine delete",
        )?;
        Ok(())
    }

    async fn can_recover(&self, vm: &VmHandle) -> Result<bool> {
        Ok(self
            .listed(&vm.id)
            .await?
            .is_some_and(|machine| machine.is_running()))
    }

    async fn is_stopped(&self, vm: &VmHandle) -> Result<bool> {
        Ok(!self.can_recover(vm).await?)
    }

    fn kind(&self) -> &'static str {
        "smolvm"
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::backend::{Ended, Stdout};

    /// A `SmolVM` whose database is locked for its first three commands.
    fn contended_smolvm(dir: &Path) -> PathBuf {
        let script = dir.join("smolvm");
        std::fs::write(
            &script,
            "#!/bin/sh\n\
             count=$(cat \"$XDG_DATA_HOME/calls\" 2>/dev/null || echo 0)\n\
             echo $((count + 1)) > \"$XDG_DATA_HOME/calls\"\n\
             if [ \"$count\" -lt 3 ]; then\n\
               echo \"Error: database operation failed: database is locked\" >&2; exit 1\n\
             fi\n\
             if [ \"$2\" = exec ]; then exit 1; fi\n\
             echo '[{\"name\":\"m\",\"cpus\":1,\"memory_mib\":256,\"state\":\"stopped\"}]'\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o755);
        std::fs::set_permissions(&script, permissions).unwrap();
        script
    }

    fn calls(dir: &Path) -> String {
        std::fs::read_to_string(dir.join("calls")).unwrap()
    }

    #[test]
    fn a_failure_is_the_error_and_not_the_progress_before_it() {
        let stderr = "Starting machine '01M7R6ES6R96MBX6EWBYH6T6BV'...\n\
                      Pulling ubuntu:no-such-tag...\n\
                      Error: image operation failed: manifest unknown\n";
        assert_eq!(failure(stderr), "image operation failed: manifest unknown");
        assert_eq!(failure("something else\n\n"), "something else");
        assert_eq!(failure(""), "");
    }

    #[tokio::test]
    async fn commands_wait_out_a_locked_database() {
        let dir = tempfile::tempdir().unwrap();
        let backend = SmolvmBackend::new(contended_smolvm(dir.path()), dir.path().to_owned());

        assert!(backend
            .is_stopped(&VmHandle { id: "m".into() })
            .await
            .unwrap());
        assert_eq!(calls(dir.path()).trim(), "4");
    }

    /// A `SmolVM` whose every command logs when it starts and ends.
    fn logging_smolvm(dir: &Path) -> PathBuf {
        let script = dir.join("smolvm");
        std::fs::write(
            &script,
            "#!/bin/sh\n\
             echo start >> \"$XDG_DATA_HOME/log\"\n\
             sleep 0.2\n\
             echo end >> \"$XDG_DATA_HOME/log\"\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o755);
        std::fs::set_permissions(&script, permissions).unwrap();
        script
    }

    #[tokio::test]
    async fn a_machines_commands_never_outnumber_its_exec_slots() {
        let dir = tempfile::tempdir().unwrap();
        let backend = SmolvmBackend::new(logging_smolvm(dir.path()), dir.path().to_owned());
        let vm = VmHandle { id: "m".into() };

        let runs =
            (0..EXEC_SLOTS * 3).map(|_| backend.run(&vm, Exec::internal("true", Stdout::Capture)));
        for out in futures::future::join_all(runs).await {
            assert_eq!(out.unwrap().ended, Ended::Exited(0));
        }

        let (mut running, mut most) = (0, 0);
        for line in std::fs::read_to_string(dir.path().join("log"))
            .unwrap()
            .lines()
        {
            running = if line == "start" {
                running + 1
            } else {
                running - 1
            };
            most = most.max(running);
        }
        assert_eq!(most, EXEC_SLOTS);
    }

    /// A `SmolVM` whose checkpoint pauses, then packs beside its output as
    /// the real one does: a temporary file first, the output at the end.
    fn packing_smolvm(dir: &Path) -> PathBuf {
        let script = dir.join("smolvm");
        std::fs::write(
            &script,
            "#!/bin/sh\n\
             for last; do :; done\n\
             sleep 0.3\n\
             touch \"$(dirname \"$last\")/.tmpPACK\"\n\
             sleep 0.6\n\
             mv \"$(dirname \"$last\")/.tmpPACK\" \"$last\"\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o755);
        std::fs::set_permissions(&script, permissions).unwrap();
        script
    }

    #[tokio::test]
    async fn a_snapshot_says_the_machine_runs_again_before_it_is_written_out() {
        let dir = tempfile::tempdir().unwrap();
        let backend = SmolvmBackend::new(packing_smolvm(dir.path()), dir.path().to_owned());
        let snap_dir = dir.path().join("snap");
        std::fs::create_dir_all(&snap_dir).unwrap();
        std::fs::write(snap_dir.join("manifest.json"), "{}").unwrap();
        let (resumed, machine_resumed) = tokio::sync::oneshot::channel();
        let started = std::time::Instant::now();

        let vm = VmHandle { id: "m".into() };
        let written = backend.snapshot_while_running(&vm, &snap_dir, resumed);
        tokio::pin!(written);
        tokio::select! {
            _ = &mut written => panic!("written out before the machine was said to run"),
            said = machine_resumed => said.unwrap(),
        }
        let resumed_after = started.elapsed();
        written.await.unwrap();

        assert!(
            resumed_after >= Duration::from_millis(300),
            "{resumed_after:?}"
        );
        assert!(started.elapsed() >= resumed_after + Duration::from_millis(400));
        assert!(snap_dir.join("machine.smolcheckpoint").exists());
    }

    #[tokio::test]
    async fn guest_commands_are_never_retried() {
        let dir = tempfile::tempdir().unwrap();
        let backend = SmolvmBackend::new(contended_smolvm(dir.path()), dir.path().to_owned());
        let vm = VmHandle { id: "m".into() };

        let out = backend
            .run(&vm, Exec::internal("true", Stdout::Capture))
            .await
            .unwrap();

        assert_eq!(out.ended, Ended::Exited(1));
        assert_eq!(calls(dir.path()).trim(), "1");
    }
}
