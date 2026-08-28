use std::{collections::HashMap, path::Path, path::PathBuf, time::Duration};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::process::Command;

use super::{RegistryAuth, Resources, RunOutput, VmBackend, VmHandle};

/// Metadata persisted into `snap_dir` so `fork_snapshot`/`resume` can find their
/// way back to the hypeman-side object a snapshot lives on. Hypeman's
/// snapshot/restore primitives are instance-scoped rather than directory-scoped
/// (there's no "restore this blob into a brand new instance" endpoint), so this
/// is the glue that lets BLAST's directory-based snapshot model sit on top of it.
#[derive(Serialize, Deserialize)]
struct SnapshotMeta {
    /// The hypeman instance this snapshot was taken from/is parked on.
    /// - After `suspend()`: this instance is in `Standby` (VMM torn down, snapshot
    ///   resident); `resume()` restores it back to `Running` in place, same id.
    /// - After `snapshot()`: this instance is still `Running`; `fork_snapshot()`
    ///   forks a brand-new instance off of it (`from_running: true`).
    instance_id: String,
    /// The specific snapshot object id, when one was minted via `POST
    /// /instances/{id}/snapshots` (i.e. from `snapshot()`, not `suspend()`).
    /// Not currently consumed by `fork_snapshot`/`resume` (both operate against
    /// `instance_id`'s live/standby state, which is what those endpoints
    /// support), but recorded for diagnostics and future use.
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot_id: Option<String>,
}

const META_FILE: &str = "hypeman.json";

pub struct HypemanBackend {
    client: Client,
    endpoint: String,
    token: String,
    cli_binary: PathBuf,
}

impl HypemanBackend {
    pub fn new(
        endpoint: impl Into<String>,
        token: impl Into<String>,
        cli_binary: impl Into<PathBuf>,
    ) -> Self {
        Self {
            client: Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            token: token.into(),
            cli_binary: cli_binary.into(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.endpoint, path.trim_start_matches('/'))
    }

    fn authed(&self, rb: RequestBuilder) -> RequestBuilder {
        rb.bearer_auth(&self.token)
    }

    fn read_meta(snap_dir: &Path) -> Result<SnapshotMeta> {
        let raw = std::fs::read_to_string(snap_dir.join(META_FILE)).with_context(|| {
            format!("no hypeman snapshot metadata in {}", snap_dir.display())
        })?;
        serde_json::from_str(&raw).context("parse hypeman snapshot metadata")
    }

    fn write_meta(snap_dir: &Path, meta: &SnapshotMeta) -> Result<()> {
        std::fs::create_dir_all(snap_dir)?;
        std::fs::write(snap_dir.join(META_FILE), serde_json::to_vec_pretty(meta)?)?;
        Ok(())
    }

    /// Block until `id` reaches `Running`, using hypeman's own `/wait` endpoint
    /// (each call is capped server-side at 5 minutes; we loop it to cover
    /// `overall_timeout`, since a single "Running" boot is normally seconds).
    async fn wait_for_running(&self, id: &str, overall_timeout: Duration) -> Result<()> {
        let deadline = std::time::Instant::now() + overall_timeout;
        loop {
            let resp: WaitResponse = self
                .authed(self.client.get(self.url(&format!("/instances/{id}/wait"))))
                .query(&[("state", "Running"), ("timeout", "10s")])
                .send()
                .await
                .context("hypeman wait")?
                .error_for_status()
                .context("hypeman wait status")?
                .json()
                .await
                .context("hypeman wait body")?;

            if resp.state == "Running" {
                return Ok(());
            }
            if !resp.timed_out {
                // The instance settled into a state that isn't Running and isn't
                // going to become Running on its own (Stopped/Unknown/etc).
                bail!(
                    "instance {id} settled into state {} instead of Running{}",
                    resp.state,
                    resp.state_error.map(|e| format!(": {e}")).unwrap_or_default(),
                );
            }
            if std::time::Instant::now() >= deadline {
                bail!("timed out waiting for instance {id} to reach Running");
            }
        }
    }

    /// Pull `image` into hypeman's image cache authenticated as `auth`, and
    /// wait for the pull+convert to finish.
    ///
    /// `POST /instances` (what `fork_image` otherwise calls directly) has no
    /// credentials field of its own -- its `credentials` field is for egress
    /// credential brokering *inside* the guest, unrelated to authenticating
    /// the base-image pull. `POST /images` (hypeman's separate "pull and
    /// convert OCI image" endpoint, also used by `hypeman pull`) is the one
    /// that takes registry credentials, via the same `PushCredentials` shape
    /// it shares with image push. So a private-image fork pulls the image
    /// there first -- authenticated, once -- and then creates the instance
    /// from what is now a locally cached, already-authenticated image
    /// reference.
    async fn pull_image_with_auth(&self, image: &str, auth: &RegistryAuth) -> Result<()> {
        let body = json!({
            "name": image,
            "credentials": {
                "username": auth.username,
                "password": auth.password,
            },
        });
        let img: HypemanImage = self
            .authed(self.client.post(self.url("/images")))
            .json(&body)
            .send()
            .await
            .context("hypeman image pull")?
            .error_for_status()
            .context("hypeman image pull status")?
            .json()
            .await
            .context("hypeman image pull body")?;
        // Poll by digest, not by `image`'s tag: hypeman only wires up the
        // tag -> digest pointer once the build finalizes (reaches `ready` or
        // `failed`), so a tag-form GET 404s the whole time it's `pending` or
        // `pulling` -- even though the record already exists and is readable
        // by digest, which createImage resolved (and returned) synchronously.
        let digest_ref = format!("{}@{}", repository_of(image), img.digest);
        self.wait_for_image_ready(&digest_ref, Duration::from_secs(300)).await
    }

    /// Poll `GET /images/{name}` (there's no `/wait` endpoint for images the
    /// way there is for instances) until the pull+convert reaches `ready` or
    /// `failed`. `name` should be a `repo@digest` reference (see
    /// [`pull_image_with_auth`]), which resolves throughout the build,
    /// unlike a `repo:tag` reference.
    async fn wait_for_image_ready(&self, name: &str, overall_timeout: Duration) -> Result<()> {
        let encoded = urlencoding::encode(name);
        let deadline = std::time::Instant::now() + overall_timeout;
        loop {
            let img: HypemanImage = self
                .authed(self.client.get(self.url(&format!("/images/{encoded}"))))
                .send()
                .await
                .context("hypeman image status")?
                .error_for_status()
                .context("hypeman image status")?
                .json()
                .await
                .context("hypeman image status body")?;
            match img.status.as_str() {
                "ready" => return Ok(()),
                "failed" => bail!(
                    "image pull failed for {name}: {}",
                    img.error.unwrap_or_else(|| "unknown error".to_owned())
                ),
                _ => {}
            }
            if std::time::Instant::now() >= deadline {
                bail!("timed out waiting for image {name} to become ready");
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }
}

/// The repository portion of an OCI image reference: everything before a
/// trailing `@digest` or `:tag`. A `:` before the final `/` is a registry
/// port, not a tag, so only the final path segment is checked for one.
fn repository_of(image: &str) -> &str {
    if let Some(at) = image.rfind('@') {
        return &image[..at];
    }
    let last_slash = image.rfind('/').map_or(0, |i| i + 1);
    if let Some(colon) = image[last_slash..].find(':') {
        return &image[..last_slash + colon];
    }
    image
}

#[derive(Deserialize)]
struct Instance {
    id: String,
}

#[derive(Deserialize)]
struct HypemanImage {
    digest: String,
    status: String,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct WaitResponse {
    state: String,
    #[serde(default)]
    state_error: Option<String>,
    timed_out: bool,
}

#[derive(Deserialize)]
struct Snapshot {
    id: String,
}

#[async_trait]
impl VmBackend for HypemanBackend {
    async fn fork_image(
        &self,
        image: &str,
        resources: &Resources,
        registry_auth: Option<&RegistryAuth>,
    ) -> Result<VmHandle> {
        if let Some(auth) = registry_auth {
            self.pull_image_with_auth(image, auth).await?;
        }
        // Instance names must match ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$; ulid's
        // Crockford-base32 rendering is uppercase, so lowercase it.
        let name = ulid::Ulid::new().to_string().to_lowercase();
        let body = json!({
            "name": name,
            "image": image,
            "vcpus": resources.vcpu,
            "size": format!("{} MB", resources.memory_mib),
            "overlay_size": format!("{} MB", resources.disk_mib),
            // Without an explicit long-running cmd, most OCI images default CMD
            // to a bare shell with no stdin attached. That shell exits (code 0)
            // within a few hundred ms of boot, which shuts the guest down and
            // makes firecracker exit cleanly right behind it -- hypeman then has
            // nothing left to query and reports the instance stuck in "Unknown"
            // forever (repeated "connection refused" on the now-gone fc.sock).
            // It isn't a firecracker crash; it's an empty entrypoint. Keep every
            // VM alive the same way DockerBackend does.
            "cmd": ["sh", "-c", "tail -f /dev/null"],
        });
        let inst: Instance = self
            .authed(self.client.post(self.url("/instances")))
            .json(&body)
            .send()
            .await
            .context("hypeman create")?
            .error_for_status()
            .context("hypeman create status")?
            .json()
            .await
            .context("hypeman create body")?;
        self.wait_for_running(&inst.id, Duration::from_secs(60)).await?;
        Ok(VmHandle { id: inst.id, platform: super::host_platform().to_owned() })
    }

    async fn fork_snapshot(&self, snap_dir: &Path, _resources: &Resources) -> Result<VmHandle> {
        // hypeman has no per-request resource override on fork: the new
        // instance inherits whatever the source was sized as.
        let meta = Self::read_meta(snap_dir)?;
        let name = ulid::Ulid::new().to_string().to_lowercase();
        let body = json!({
            "name": name,
            // Safe to send unconditionally: it only changes behavior when the
            // source instance actually is Running (pause, fork, resume it back);
            // a Standby/Stopped source ignores it.
            "from_running": true,
            "target_state": "Running",
        });
        let inst: Instance = self
            .authed(self.client.post(self.url(&format!("/instances/{}/fork", meta.instance_id))))
            .json(&body)
            .send()
            .await
            .context("hypeman fork")?
            .error_for_status()
            .context("hypeman fork status")?
            .json()
            .await
            .context("hypeman fork body")?;
        self.wait_for_running(&inst.id, Duration::from_secs(60)).await?;
        Ok(VmHandle { id: inst.id, platform: super::host_platform().to_owned() })
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
        // There's no plain REST exec endpoint (it's WebSocket-only, driven by
        // the hypeman-cli); shell out to the real CLI instead of reimplementing
        // that protocol.
        let mut cmd = Command::new(&self.cli_binary);
        cmd.env("HYPEMAN_BASE_URL", &self.endpoint);
        cmd.env("HYPEMAN_API_KEY", &self.token);
        cmd.args(["exec", &vm.id, "--cwd", cwd, "--timeout", &timeout.as_secs().to_string()]);
        for (k, v) in env {
            cmd.arg("-e").arg(format!("{k}={v}"));
        }
        cmd.args(["--", "sh", "-c", command]);

        let out = tokio::time::timeout(timeout, cmd.output())
            .await
            .map_err(|_| anyhow!("hypeman exec timed out"))?
            .context("hypeman exec")?;
        Ok(RunOutput {
            stdout: out.stdout,
            stderr: out.stderr,
            exit_code: out.status.code().unwrap_or(-1),
        })
    }

    // Hypeman has no CPU-only freeze primitive over this API: `standby` always
    // pairs pause with a snapshot + VMM teardown, which is `suspend()`'s job,
    // not `pause()`'s. Same call smolvm.rs makes, and for the same reason.
    async fn pause(&self, _vm: &VmHandle) -> Result<()> {
        Ok(())
    }

    async fn unpause(&self, _vm: &VmHandle) -> Result<()> {
        Ok(())
    }

    async fn snapshot(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()> {
        let snap: Snapshot = self
            .authed(self.client.post(self.url(&format!("/instances/{}/snapshots", vm.id))))
            .json(&json!({ "kind": "Standby" }))
            .send()
            .await
            .context("hypeman snapshot")?
            .error_for_status()
            .context("hypeman snapshot status")?
            .json()
            .await
            .context("hypeman snapshot body")?;
        Self::write_meta(
            snap_dir,
            &SnapshotMeta { instance_id: vm.id.clone(), snapshot_id: Some(snap.id) },
        )
    }

    async fn suspend(&self, vm: &VmHandle, snap_dir: &Path) -> Result<()> {
        // standby = pause + snapshot + delete VMM, in one call, in place. The
        // instance record (and its snapshot) survive under the same id, which
        // is exactly what resume()/fork_snapshot() need to find later.
        self.authed(self.client.post(self.url(&format!("/instances/{}/standby", vm.id))))
            .json(&json!({}))
            .send()
            .await
            .context("hypeman standby")?
            .error_for_status()
            .context("hypeman standby status")?;
        Self::write_meta(snap_dir, &SnapshotMeta { instance_id: vm.id.clone(), snapshot_id: None })
    }

    async fn resume(&self, snap_dir: &Path, _resources: &Resources) -> Result<VmHandle> {
        // hypeman has no per-request resource override on restore either: it
        // restores the standby instance back to exactly what it was.
        let meta = Self::read_meta(snap_dir)?;
        self.authed(self.client.post(self.url(&format!("/instances/{}/restore", meta.instance_id))))
            .send()
            .await
            .context("hypeman restore")?
            .error_for_status()
            .context("hypeman restore status")?;
        self.wait_for_running(&meta.instance_id, Duration::from_secs(60)).await?;
        Ok(VmHandle { id: meta.instance_id, platform: super::host_platform().to_owned() })
    }

    async fn delete(&self, vm: &VmHandle) -> Result<()> {
        self.authed(self.client.delete(self.url(&format!("/instances/{}", vm.id))))
            .send()
            .await
            .context("hypeman delete")?
            .error_for_status()
            .context("hypeman delete status")?;
        Ok(())
    }

    fn platform(&self) -> &str {
        super::host_platform()
    }
}
