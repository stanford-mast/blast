use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::config::BackendConfig;

pub const SMOLVM_RELEASE_VERSION: &str = "1.13.3";

pub fn ensure_backend_ready(cfg: &mut BackendConfig, _data_dir: &Path) -> Result<()> {
    match cfg {
        BackendConfig::Docker => {
            which::which("docker").context("docker is not installed or not on PATH")?;
        }
        BackendConfig::Smolvm { binary } => {
            check_kvm_available()?;
            *binary = resolve(binary, "smolvm")?;
        }
        BackendConfig::Hypeman {
            endpoint,
            token,
            cli_binary,
        } => {
            if endpoint.trim().is_empty() {
                bail!("hypeman endpoint is required");
            }
            if token.trim().is_empty() {
                bail!("hypeman token is required");
            }
            *cli_binary = resolve(cli_binary, "hypeman")?;
        }
    }
    Ok(())
}

fn resolve(binary: &Path, name: &str) -> Result<PathBuf> {
    which::which(binary).with_context(|| {
        format!("{name} is not installed or not on PATH; configure an explicit binary path")
    })
}

#[cfg(target_os = "linux")]
fn check_kvm_available() -> Result<()> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kvm")
        .context("smolvm requires readable and writable /dev/kvm")?;
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn check_kvm_available() -> Result<()> {
    Ok(())
}
