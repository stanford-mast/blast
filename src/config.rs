use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default)]
    pub backend: BackendConfig,

    #[serde(default)]
    pub worker: WorkerConfig,

    #[serde(default)]
    pub lifecycle: LifecycleConfig,

    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
}

// Every field below carries its own `#[serde(default)]`, not just the
// `impl Default for BackendConfig` further down. The whole-enum default only
// applies when the *entire* `backend` table is absent (e.g. no `[backend]`
// section in TOML at all); as soon as any key under it is present -- e.g.
// just `BLAST__BACKEND__KIND=hypeman`, with no endpoint/token, so that
// `backend.bootstrap` can auto-provision the rest -- serde deserializes a
// concrete `BackendConfig` variant from what's there, and every field it
// doesn't find still needs a fallback or deserialization fails outright with
// "missing field". Per-field defaults are what make `kind`-only overrides
// work for every variant, not just whichever one happens to be selected by
// omitting `[backend]` entirely.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BackendConfig {
    Docker,
    Hypeman {
        /// Defaults to a local hypeman on its conventional port. When this
        /// isn't reachable at startup, `backend::bootstrap` auto-builds and
        /// spawns one right here rather than treating it as an error.
        #[serde(default = "default_hypeman_endpoint")]
        endpoint: String,
        /// Bearer token for the hypeman REST API and CLI alike (a hypeman JWT,
        /// e.g. minted via hypeman's `gen-jwt` tool). Empty by default: when
        /// `backend::bootstrap` auto-spawns a local hypeman it mints one and
        /// fills this in; if you're pointing at an already-running instance
        /// instead, set this (an empty token gets every request rejected).
        #[serde(default)]
        token: String,
        /// Path to the `hypeman` CLI binary, used for `run()` since exec has no
        /// plain REST endpoint (it's WebSocket-only). Bare `"hypeman"` resolves
        /// via `PATH`, same as `"docker"`/`"smolvm"` already do for their backends.
        #[serde(default = "default_hypeman_cli_binary")]
        cli_binary: PathBuf,
    },
    Smolvm {
        /// Bare `"smolvm"` resolves via `PATH`; `backend::bootstrap` falls
        /// back to a BLAST-managed, auto-built copy when that lookup misses.
        #[serde(default = "default_smolvm_binary")]
        binary: PathBuf,
    },
}

impl Default for BackendConfig {
    /// `SmolVM` (libkrun-based microVMs) is the primary backend: real
    /// hardware-isolated memory snapshotting via fork/snapshot/restore,
    /// unlike Docker's commit-based pseudo-snapshot. A bare `"smolvm"`
    /// resolves via `PATH`, same as `"docker"` already does for `DockerBackend`.
    fn default() -> Self {
        Self::Smolvm { binary: default_smolvm_binary() }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkerResources {
    pub vcpu: u32,
    pub memory_mib: u64,
    pub disk_mib: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkerConfig {
    pub control_plane_endpoint: Option<String>,
    pub api_key: Option<String>,
    /// Single-use registration token (wrt_...) included in the first register request.
    pub registration_token: Option<String>,
    /// Provider label advertised on registration (default: "blast").
    #[serde(default = "default_provider")]
    pub provider: String,
    pub region: Option<String>,
    /// Total resource pool this worker makes available for VMs + snapshots.
    /// Running and paused VMs hold vcpu + memory; all states hold disk.
    /// If absent, the worker registers with zero capacity upstream
    /// (standalone mode) -- and, just as importantly, `handle_fork` skips
    /// local admission control entirely (unlimited, matching pre-existing
    /// behavior for dev/standalone use). Set this to get BOTH: upstream
    /// advertises real capacity, AND `fork` is admission-controlled and
    /// queues locally against it, on every backend uniformly.
    pub resources: Option<WorkerResources>,
    /// How long a fork request waits for pool headroom to free up (via a
    /// VM being deleted/paused/suspended elsewhere) before giving up, when
    /// `resources` is configured and momentarily exhausted. A request whose
    /// own size exceeds `resources` outright fails immediately regardless
    /// of this -- no amount of waiting ever makes it fit. Ignored when
    /// `resources` is absent.
    #[serde(default = "default_admission_queue_secs")]
    pub admission_queue_secs: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            control_plane_endpoint: None,
            api_key: None,
            registration_token: None,
            provider: default_provider(),
            region: None,
            resources: None,
            admission_queue_secs: default_admission_queue_secs(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LifecycleConfig {
    /// Idle seconds before pausing a VM (frees CPU; memory stays hot).
    #[serde(default = "default_pause_ttl")]
    pub pause_ttl_secs: u64,

    /// Idle seconds before suspending a paused VM (frees CPU + memory; snapshot kept on disk).
    #[serde(default = "default_suspend_ttl")]
    pub suspend_ttl_secs: u64,

    /// Idle seconds before evicting a suspended VM (frees CPU + memory + disk).
    #[serde(default = "default_evict_ttl")]
    pub evict_ttl_secs: u64,

    /// Seconds between dirty-sync snapshots of running VMs.
    #[serde(default = "default_dirty_sync_ttl")]
    pub dirty_sync_ttl_secs: u64,

    /// Fraction of total memory below which suspended VMs are evicted proactively.
    #[serde(default = "default_memory_pressure_thresh")]
    pub memory_pressure_thresh: f64,

    /// Fraction of total disk below which suspended VMs are evicted proactively.
    #[serde(default = "default_disk_pressure_thresh")]
    pub disk_pressure_thresh: f64,

    /// Fraction of total vcpu below which running VMs are eagerly paused
    /// proactively (before their normal `pause_ttl` would fire). Also
    /// checked, alongside a nonzero pool admission-waiter count, on every
    /// pressure-loop pass: a fork parked waiting for capacity makes this
    /// check fire regardless of the ratio, same as the memory/disk checks.
    #[serde(default = "default_vcpu_pressure_thresh")]
    pub vcpu_pressure_thresh: f64,

    /// Maximum snapshot storage in MiB before LRU eviction.
    #[serde(default = "default_max_snapshot_disk_mib")]
    #[allow(dead_code)]
    pub max_snapshot_disk_mib: u64,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            pause_ttl_secs: default_pause_ttl(),
            suspend_ttl_secs: default_suspend_ttl(),
            evict_ttl_secs: default_evict_ttl(),
            dirty_sync_ttl_secs: default_dirty_sync_ttl(),
            memory_pressure_thresh: default_memory_pressure_thresh(),
            disk_pressure_thresh: default_disk_pressure_thresh(),
            vcpu_pressure_thresh: default_vcpu_pressure_thresh(),
            max_snapshot_disk_mib: default_max_snapshot_disk_mib(),
        }
    }
}

const fn default_port() -> u16 { 7240 }
fn default_data_dir() -> PathBuf { PathBuf::from("./blast-data") }
fn default_hypeman_cli_binary() -> PathBuf { PathBuf::from("hypeman") }
fn default_hypeman_endpoint() -> String { "http://127.0.0.1:4973".into() }
fn default_smolvm_binary() -> PathBuf { PathBuf::from("smolvm") }
fn default_provider() -> String { "blast".into() }
const fn default_admission_queue_secs() -> u64 { 120 }
const fn default_pause_ttl() -> u64 { 60 }
const fn default_suspend_ttl() -> u64 { 300 }
const fn default_evict_ttl() -> u64 { 3600 }
const fn default_dirty_sync_ttl() -> u64 { 60 }
const fn default_memory_pressure_thresh() -> f64 { 0.15 }
const fn default_disk_pressure_thresh() -> f64 { 0.10 }
const fn default_vcpu_pressure_thresh() -> f64 { 0.15 }
const fn default_max_snapshot_disk_mib() -> u64 { 50_000 }

impl Config {
    pub fn load(config_path: Option<&std::path::Path>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();
        if let Some(path) = config_path {
            builder = builder.add_source(config::File::from(path));
        } else {
            builder = builder.add_source(config::File::with_name("blast").required(false));
        }
        let cfg = builder
            .add_source(config::Environment::with_prefix("BLAST").separator("__"))
            .build()?
            .try_deserialize()?;
        Ok(cfg)
    }
}
