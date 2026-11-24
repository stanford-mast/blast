use std::{path::Path, process::Stdio};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use tokio::process::Command;

use super::{Exec, RegistryAuth, Resources, RunOutput, VmBackend, VmHandle};

pub struct DockerBackend;

impl DockerBackend {
    pub const fn new() -> Self {
        Self
    }

    async fn exec_docker(args: &[&str]) -> Result<std::process::Output> {
        Command::new("docker")
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("docker exec")
    }

    /// Pull `image`, authenticated against a private registry via a
    /// per-invocation `DOCKER_CONFIG` directory scoped to just this one pull.
    ///
    /// A plain `docker login` writes into the shared, global
    /// `~/.docker/config.json` -- fine for a single interactive user, but a
    /// race under concurrent forks that pull from different registries (or
    /// the same registry with different credentials) at once. Pointing
    /// `DOCKER_CONFIG` at a fresh, single-registry directory for just this
    /// `docker pull` subprocess avoids mutating any shared state at all. The
    /// directory (and the credential inside it) is removed immediately after
    /// the pull, whether it succeeded or failed.
    async fn pull_with_auth(image: &str, auth: &RegistryAuth) -> Result<()> {
        let registry = super::extract_registry(image);
        // Docker's own credential store keys Docker Hub under this canonical
        // URL rather than the "docker.io" alias `docker login` also accepts.
        let key = if registry == "docker.io" {
            "https://index.docker.io/v1/"
        } else {
            registry
        };
        let auth_b64 = B64.encode(format!("{}:{}", auth.username, auth.password));
        let config = serde_json::json!({ "auths": { key: { "auth": auth_b64 } } });

        let dir = std::env::temp_dir().join(format!("blast-docker-auth-{}", ulid::Ulid::new()));
        tokio::fs::create_dir_all(&dir).await?;
        let write_result =
            tokio::fs::write(dir.join("config.json"), serde_json::to_vec(&config)?).await;

        let pull_result = match write_result {
            Ok(()) => Command::new("docker")
                .env("DOCKER_CONFIG", &dir)
                .args(["pull", image])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
                .context("docker pull"),
            Err(e) => Err(e).context("write scoped docker credential file"),
        };

        // Never leave the scoped credential behind longer than this one pull
        // needed it, regardless of whether the pull itself succeeded.
        let _ = tokio::fs::remove_dir_all(&dir).await;

        let out = pull_result?;
        if !out.status.success() {
            bail!(
                "docker pull failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Ok(())
    }
}

#[async_trait]
impl VmBackend for DockerBackend {
    async fn fork_image(
        &self,
        image: &str,
        _resources: &Resources,
        registry_auth: Option<&RegistryAuth>,
    ) -> Result<VmHandle> {
        if let Some(auth) = registry_auth {
            Self::pull_with_auth(image, auth).await?;
        }
        let out = Self::exec_docker(&["run", "-d", "--rm", image, "sh", "-c", "tail -f /dev/null"])
            .await?;
        if !out.status.success() {
            bail!(
                "docker run failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let id = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        Ok(VmHandle { id })
    }

    async fn fork_snapshot(&self, snap_dir: &Path, _resources: &Resources) -> Result<VmHandle> {
        // Restore from a tar.gz snapshot written by snapshot() or suspend().
        let snap = snap_dir.join("container.tar.gz");
        if !snap.exists() {
            bail!("no container snapshot at {}", snap.display());
        }
        let out = Command::new("docker")
            .args(["load", "-i", &snap.to_string_lossy()])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;
        if !out.status.success() {
            bail!(
                "docker load failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let load_output = String::from_utf8_lossy(&out.stdout);
        let image_name = load_output
            .lines()
            .find_map(|line| {
                line.strip_prefix("Loaded image: ")
                    .or_else(|| line.strip_prefix("Loaded image ID: "))
            })
            .ok_or_else(|| {
                anyhow!("docker load did not print a loaded image name; output: {load_output}")
            })?
            .trim()
            .to_owned();
        let out = Command::new("docker")
            .args([
                "run",
                "-d",
                "--rm",
                &image_name,
                "sh",
                "-c",
                "tail -f /dev/null",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;
        if !out.status.success() {
            bail!(
                "docker run failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let id = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        Ok(VmHandle { id })
    }

    async fn run(&self, vm: &VmHandle, exec: Exec<'_>) -> Result<RunOutput> {
        let mut command = Command::new("docker");
        command.arg("exec");
        for (key, value) in exec.env {
            command.args(["-e", &format!("{key}={value}")]);
        }
        if let Some(cwd) = exec.cwd {
            command.args(["-w", cwd]);
        }
        command.args([&vm.id, "sh", "-c", exec.command]);
        super::exec_client(command, &exec).await
    }

    async fn pause(&self, vm: &VmHandle) -> Result<()> {
        let out = Self::exec_docker(&["pause", &vm.id]).await?;
        if !out.status.success() {
            bail!(
                "docker pause failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Ok(())
    }

    async fn unpause(&self, vm: &VmHandle) -> Result<()> {
        let out = Self::exec_docker(&["unpause", &vm.id]).await?;
        if !out.status.success() {
            bail!(
                "docker unpause failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Ok(())
    }

    fn supports_pause(&self) -> bool {
        true
    }

    async fn snapshot(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()> {
        let snap_tag = format!("blast_snap_{}", vm.id);
        // Commit the running container state to a named image.
        let out = Command::new("docker")
            .args(["commit", &vm.id, &snap_tag])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;
        if !out.status.success() {
            bail!(
                "docker commit failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        // Save and gzip the image into snap_dir. The container keeps running.
        tokio::fs::create_dir_all(snap_dir).await?;
        // Replace, never rewrite, the previous snapshot: a fork may still be
        // reading it.
        let partial = snap_dir.join("container.tar.gz.partial");
        let out = Command::new("sh")
            .arg("-c")
            .arg(format!(
                "docker save {} | gzip > {}",
                snap_tag,
                partial.display()
            ))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;
        if !out.status.success() {
            bail!(
                "docker save failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        tokio::fs::rename(&partial, snap_dir.join("container.tar.gz")).await?;
        // Remove the named image to avoid accumulation.
        Command::new("docker")
            .args(["rmi", "-f", &snap_tag])
            .output()
            .await
            .ok();
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
        // Best-effort; ignore error if already removed.
        Self::exec_docker(&["rm", "-f", &vm.id]).await?;
        Ok(())
    }

    async fn write_file(
        &self,
        vm: &VmHandle,
        guest_path: &str,
        src: &Path,
        mode: u32,
    ) -> Result<()> {
        // The path and mode travel as positional arguments, so they need no quoting.
        const SCRIPT: &str = r#"if [ -d "$1" ]; then echo "$3" >&2; exit 21; fi
mkdir -p "$(dirname "$1")" && cat > "$1.blast-partial" && chmod "$2" "$1.blast-partial" \
    && mv -f "$1.blast-partial" "$1" || { rm -f "$1.blast-partial"; exit 1; }"#;
        let file = std::fs::File::open(src).context("open staged sync file")?;
        let mode = format!("{mode:o}");
        let out = Command::new("docker")
            .args(["exec", "-i", &vm.id, "sh", "-c", SCRIPT, "sh", guest_path])
            .args([&mode, super::IS_A_DIRECTORY])
            .stdin(Stdio::from(file))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("docker exec")?;
        if !out.status.success() {
            bail!(
                "write {guest_path}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(())
    }

    fn kind(&self) -> &'static str {
        "docker"
    }
}
