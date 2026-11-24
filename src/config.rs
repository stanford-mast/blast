use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default)]
    pub backend: BackendConfig,

    #[serde(default)]
    pub api: ApiConfig,

    #[serde(default)]
    pub worker: WorkerConfig,

    #[serde(default)]
    pub lifecycle: LifecycleConfig,

    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    /// Total resources available to VMs and snapshots.
    pub resources: Option<ResourcesConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BackendConfig {
    Docker,
    Hypeman {
        /// Hypeman API endpoint.
        #[serde(default = "default_hypeman_endpoint")]
        endpoint: String,
        /// Bearer token for the Hypeman API and CLI.
        #[serde(default)]
        token: String,
        /// Path to the `hypeman` CLI binary.
        #[serde(default = "default_hypeman_cli_binary")]
        cli_binary: PathBuf,
    },
    Smolvm {
        /// Bare `"smolvm"` resolves via `PATH`.
        #[serde(default = "default_smolvm_binary")]
        binary: PathBuf,
    },
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self::Smolvm {
            binary: default_smolvm_binary(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourcesConfig {
    pub vcpu: u32,
    pub memory_mib: u64,
    pub disk_mib: u64,
    /// Maximum time a fork waits for capacity.
    #[serde(default = "default_admission_queue_secs")]
    pub admission_queue_secs: u64,
}

/// Remote control-plane configuration. Without it, BLAST uses its embedded
/// local control plane.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerConfig {
    /// The control plane's regional endpoint, e.g. `https://control-plane.example.com`.
    pub control_plane_endpoint: Option<String>,
    /// The pool's join key (scope `pool:<uuid>:join`).
    pub api_key: Option<String>,
    pub pool_id: Option<String>,
    /// Optional compatibility suffix (for example a required CPU baseline).
    /// BLAST always prefixes it with snapshot ABI, backend, host OS and ISA.
    pub platform: Option<String>,
    /// Upper bound on how long stopping or leaving waits for in-flight runs,
    /// and again for snapshot copies, before the worker exits.
    #[serde(default = "default_drain_timeout_secs")]
    pub drain_timeout_secs: u64,
    /// Optional durable snapshot directory.
    pub storage_dir: Option<PathBuf>,
    /// Whether `storage_dir` is shared by every pool worker.
    #[serde(default)]
    pub storage_dir_shared: bool,
    /// S3-compatible endpoint for that object storage (path-style), e.g. a
    /// `MinIO` server. Unset means AWS S3 in the bucket's region.
    pub object_storage_endpoint: Option<String>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            control_plane_endpoint: None,
            api_key: None,
            pool_id: None,
            platform: None,
            drain_timeout_secs: default_drain_timeout_secs(),
            storage_dir: None,
            storage_dir_shared: false,
            object_storage_endpoint: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RemoteConfig {
    pub endpoint: String,
    pub api_key: String,
    pub pool_id: String,
    pub drain_timeout_secs: u64,
}

impl WorkerConfig {
    pub fn remote(&self) -> anyhow::Result<Option<RemoteConfig>> {
        let Some(endpoint) = &self.control_plane_endpoint else {
            return Ok(None);
        };
        let (Some(api_key), Some(pool_id)) = (&self.api_key, &self.pool_id) else {
            anyhow::bail!("[worker] control_plane_endpoint requires api_key and pool_id");
        };
        Ok(Some(RemoteConfig {
            endpoint: endpoint.trim_end_matches('/').to_owned(),
            api_key: api_key.clone(),
            pool_id: pool_id.clone(),
            drain_timeout_secs: self.drain_timeout_secs,
        }))
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ApiConfig {
    /// When set, the local listener requires `Authorization: Bearer <token>`
    /// (except on `/v1/live`). Requests arriving over the worker tunnel are
    /// not checked: only the router can write to it.
    pub token: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LifecycleConfig {
    /// Seconds between durable checkpoints of running portable VMs. Zero
    /// disables dirty checkpoints.
    #[serde(default = "default_dirty_checkpoint_interval")]
    pub dirty_checkpoint_interval_secs: u64,
    /// Idle seconds before pausing a VM (frees CPU; memory stays hot).
    #[serde(default = "default_pause_ttl")]
    pub pause_ttl_secs: u64,

    /// Idle seconds before suspending a paused VM (frees CPU + memory; snapshot kept on disk).
    #[serde(default = "default_suspend_ttl")]
    pub suspend_ttl_secs: u64,

    /// Fraction of total memory below which paused VMs are suspended proactively.
    #[serde(default = "default_memory_pressure_thresh")]
    pub memory_pressure_thresh: f64,

    /// Fraction of total vcpu below which running VMs are eagerly paused
    /// proactively (before their normal `pause_ttl` would fire). Also
    /// checked, alongside a nonzero pool admission-waiter count, on every
    /// pressure-loop pass: a fork parked waiting for capacity makes this
    /// check fire regardless of the ratio, same as the memory/disk checks.
    #[serde(default = "default_vcpu_pressure_thresh")]
    pub vcpu_pressure_thresh: f64,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            dirty_checkpoint_interval_secs: default_dirty_checkpoint_interval(),
            pause_ttl_secs: default_pause_ttl(),
            suspend_ttl_secs: default_suspend_ttl(),
            memory_pressure_thresh: default_memory_pressure_thresh(),
            vcpu_pressure_thresh: default_vcpu_pressure_thresh(),
        }
    }
}

const fn default_port() -> u16 {
    7240
}
fn default_data_dir() -> PathBuf {
    PathBuf::from("./blast-data")
}
fn default_hypeman_cli_binary() -> PathBuf {
    PathBuf::from("hypeman")
}
fn default_hypeman_endpoint() -> String {
    "http://127.0.0.1:4973".into()
}
fn default_smolvm_binary() -> PathBuf {
    PathBuf::from("smolvm")
}
const fn default_drain_timeout_secs() -> u64 {
    300
}
const fn default_admission_queue_secs() -> u64 {
    120
}
const fn default_pause_ttl() -> u64 {
    60
}
const fn default_suspend_ttl() -> u64 {
    300
}
const fn default_dirty_checkpoint_interval() -> u64 {
    60
}
const fn default_memory_pressure_thresh() -> f64 {
    0.15
}
const fn default_vcpu_pressure_thresh() -> f64 {
    0.15
}

impl Config {
    pub fn load(config_path: Option<&std::path::Path>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();
        if let Some(path) = config_path {
            builder = builder.add_source(config::File::from(path));
        } else {
            builder = builder.add_source(config::File::with_name("blast").required(false));
        }
        let mut cfg: Self = builder
            .add_source(config::Environment::with_prefix("BLAST").separator("__"))
            .build()?
            .try_deserialize()?;
        // Backends are handed this directory in their environment, where a
        // relative path does not mean the same place (the XDG variables
        // SmolVM reads are ignored unless absolute).
        cfg.data_dir = std::path::absolute(&cfg.data_dir)?;
        Ok(cfg)
    }

    /// Ignored when `resources` is absent, since there's no pool to wait on.
    pub fn admission_queue_secs(&self) -> u64 {
        self.resources
            .as_ref()
            .map_or(default_admission_queue_secs(), |r| r.admission_queue_secs)
    }

    pub fn validate_resources(&self, connected: bool) -> anyhow::Result<()> {
        if !connected {
            return Ok(());
        }
        let resources = self.resources.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "connected workers require explicit [resources] vcpu, memory_mib, and disk_mib"
            )
        })?;
        if resources.vcpu == 0 || resources.memory_mib == 0 || resources.disk_mib == 0 {
            anyhow::bail!(
                "connected worker [resources] vcpu, memory_mib, and disk_mib must all be nonzero"
            );
        }
        Ok(())
    }
}
