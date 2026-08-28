use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use tokio::process::Command;

use super::{RegistryAuth, Resources, RunOutput, VmBackend, VmHandle};

/// The subset of `smolvm machine ls --json` this backend reads back
/// after a checkpoint restore, to confirm what topology it actually got
/// (`create --from` itself takes no sizing flags to echo).
#[derive(serde::Deserialize)]
struct MachineInfo {
    name: String,
    cpus: u32,
    memory_mib: u64,
}

pub struct SmolvmBackend {
    binary: std::path::PathBuf,
    /// Rooted under BLAST's own configured data_dir, so smolvm's VM images,
    /// checkpoints, and server DB land wherever the operator put BLAST's
    /// data, not smolvm's own default (XDG_DATA_HOME / ~/.local/share),
    /// which is silently independent of BLAST's config otherwise and, on a
    /// host with a small root disk and a separate data volume, fills root
    /// regardless of how data_dir is set.
    smolvm_data_dir: std::path::PathBuf,
}

/// Escape a string for embedding in a TOML basic string (`"..."`).
fn toml_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

impl SmolvmBackend {
    pub fn new(binary: std::path::PathBuf, data_dir: std::path::PathBuf) -> Self {
        Self { binary, smolvm_data_dir: data_dir }
    }

    async fn smolvm(&self, args: &[&str]) -> Result<std::process::Output> {
        self.smolvm_env(args, None).await
    }

    /// Like [`Self::smolvm`], but with `SMOLVM_CONFIG` pointed at
    /// `config_path` when given, instead of smolvm's default
    /// `~/.config/smolvm/config.toml`.
    async fn smolvm_env(
        &self,
        args: &[&str],
        config_path: Option<&Path>,
    ) -> Result<std::process::Output> {
        let mut cmd = Command::new(&self.binary);
        // Matches build-agent-rootfs.sh --install (bootstrap.rs), which
        // reads $XDG_DATA_HOME directly rather than smolvm's own
        // SMOLVM_DATA_DIR (that path additionally relocates HOME and removes
        // XDG_DATA_HOME, which double-nests to data_dir/smolvm/.local/share/
        // smolvm instead of data_dir/smolvm -- confirmed the hard way).
        // dirs::data_local_dir() reads XDG_DATA_HOME natively, so this alone
        // is what actually relocates VM images/checkpoints/server DB under
        // BLAST's data_dir, not just BLAST's own bookkeeping.
        cmd.env("XDG_DATA_HOME", &self.smolvm_data_dir);
        // Checkpoint staging (portable_checkpoint.rs::staging_root) has its
        // own explicit SMOLVM_PACK_STAGING override, checked before it ever
        // falls back to dirs::cache_dir() -- set for precision, since this is
        // the one cache path we know by name.
        cmd.env("SMOLVM_PACK_STAGING", self.smolvm_data_dir.join("pack-staging"));
        // Everything else that's dirs::cache_dir()-based (confirmed the hard
        // way: VM instance directories -- disk images, boot configs -- kept
        // landing under $HOME/.cache/smolvm/vms even after both fixes above)
        // goes through this broader XDG_CACHE_HOME instead, since this
        // backend hasn't traced every remaining call site by name.
        cmd.env("XDG_CACHE_HOME", self.smolvm_data_dir.join("cache"));
        if let Some(p) = config_path {
            cmd.env("SMOLVM_CONFIG", p);
        }
        cmd.args(args).output().await.map_err(Into::into)
    }

    fn check(out: std::process::Output, ctx: &str) -> Result<std::process::Output> {
        if out.status.success() {
            Ok(out)
        } else {
            bail!("{ctx}: {}", String::from_utf8_lossy(&out.stderr))
        }
    }

    /// Write a `SMOLVM_CONFIG`-scoped registry config file (see smolvm's own
    /// `SMOLVM_CONFIG` override, meant for exactly this: "CI/CD and server
    /// deployments where the config file is placed at a non-standard
    /// location") containing only `image`'s registry's credentials, in a
    /// fresh temp directory.
    ///
    /// Scoping credentials per-invocation, rather than writing into the
    /// shared `~/.config/smolvm/config.toml` (what `smolvm config registries
    /// edit` would mutate), means concurrent forks with different
    /// credentials never race on shared global state. Returns the path to
    /// the written `config.toml`; the caller is responsible for removing its
    /// parent directory once the pull it was needed for is done.
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
        let out = Self::check(self.smolvm(&["machine", "ls", "--json"]).await?, "smolvm machine ls")?;
        let list: Vec<MachineInfo> = serde_json::from_slice(&out.stdout)
            .context("parse smolvm machine ls --json")?;
        list.into_iter()
            .find(|m| m.name == name)
            .ok_or_else(|| anyhow::anyhow!("smolvm machine ls: {name} not found after create"))
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
                        "machine", "create",
                        "--name", &id,
                        "--image", image,
                        "--net",
                        "--cpus", &resources.vcpu.to_string(),
                        "--mem", &mem,
                        "--storage", &storage_gib,
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
        run_result?;
        Ok(VmHandle { id, platform: super::host_platform().to_owned() })
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
                "machine", "create",
                "--name", &id,
                "--from", &checkpoint_path.to_string_lossy(),
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
        if restored.cpus != resources.vcpu || restored.memory_mib != resources.memory_mib {
            let _ = self.smolvm(&["machine", "delete", "--name", &id, "--force"]).await;
            bail!(
                "cannot resize on fork from a running VM: source is {}vcpu/{}MiB, requested {}vcpu/{}MiB. SmolVM's live-checkpoint restore always inherits the source's exact topology; omit `resources` to inherit it, or fork from an image instead if a specific size is required.",
                restored.cpus, restored.memory_mib, resources.vcpu, resources.memory_mib
            );
        }
        Self::check(
            self.smolvm(&["machine", "start", "--name", &id]).await?,
            "smolvm start",
        )?;
        Ok(VmHandle { id, platform: super::host_platform().to_owned() })
    }


    async fn run(
        &self,
        vm: &VmHandle,
        command: &str,
        _session_id: &str,
        env: &HashMap<String, String>,
        cwd: &str,
        timeout: Duration,
    ) -> Result<RunOutput> {
        let timeout_arg = format!("{}s", timeout.as_secs());
        let mut args: Vec<&str> = vec![
            "machine", "exec",
            "--name", &vm.id,
            "--workdir", cwd,
            "--timeout", &timeout_arg,
        ];
        let env_args: Vec<String> = env.iter().map(|(k, v)| format!("{k}={v}")).collect();
        for e in &env_args {
            args.push("--env");
            args.push(e);
        }
        args.push("--");
        args.push("sh");
        args.push("-c");
        args.push(command);

        let out = tokio::time::timeout(timeout, self.smolvm(&args))
            .await
            .map_err(|_| anyhow::anyhow!("exec timed out"))??;
        Ok(RunOutput {
            stdout: out.stdout,
            stderr: out.stderr,
            exit_code: out.status.code().unwrap_or(-1),
        })
    }

    // smolvm has no distinct pause; fall through to suspend semantics.
    async fn pause(&self, _vm: &VmHandle) -> Result<()> {
        // No-op: smolvm doesn't expose a separate CPU-freeze primitive.
        // The caller will follow up with suspend() when the suspend TTL fires.
        Ok(())
    }

    async fn unpause(&self, _vm: &VmHandle) -> Result<()> {
        Ok(())
    }

    async fn snapshot(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(snap_dir)?;
        let checkpoint_path = snap_dir.join("machine.smolcheckpoint");
        Self::check(
            self.smolvm(&[
                "machine", "checkpoint",
                "--name", &vm.id,
                "--output", &checkpoint_path.to_string_lossy(),
            ])
            .await?,
            "smolvm checkpoint",
        )?;
        Ok(())
    }

    async fn suspend(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()> {
        self.snapshot(vm, snap_dir).await?;
        self.delete(vm).await
    }

    async fn resume(&self, snap_dir: &Path, resources: &Resources) -> Result<VmHandle> {
        self.fork_snapshot(snap_dir, resources).await
    }

    async fn delete(&self, vm: &VmHandle) -> Result<()> {
        self.smolvm(&["machine", "delete", "--name", &vm.id, "--force"]).await?;
        Ok(())
    }

    fn platform(&self) -> &str { super::host_platform() }
}
