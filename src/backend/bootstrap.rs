//! Zero-config backend runtime bootstrap.
//!
//! BLAST is meant to run with no external setup: point it at nothing and it
//! should auto-detect and auto-provision whatever the configured backend
//! needs. This module resolves that *before* `main` constructs the
//! `VmBackend` trait object, mutating the loaded [`BackendConfig`] in place
//! (e.g. filling in a bare `"smolvm"` with the absolute path of a binary it
//! just built, or a hypeman `token` it just minted) so backend construction
//! downstream never has to know whether a value came from the operator or
//! from us.
//!
//! Docker is intentionally left alone: like the kernel, it's assumed to
//! already be present as host infrastructure.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, BufReader};
use tokio::process::Command;
use tracing::{debug, info, warn};

use crate::config::BackendConfig;

/// Managed binaries/build artifacts live under `<data_dir>/bin`, alongside
/// `<data_dir>/snapshots` (see `SnapshotStore::snap_dir`) -- `data_dir` is
/// already the one place BLAST owns on disk, so we don't invent a second one.
fn managed_bin(data_dir: &Path, name: &str) -> PathBuf {
    data_dir.join("bin").join(name)
}

fn build_scratch_dir(data_dir: &Path, name: &str) -> PathBuf {
    data_dir.join(".build").join(format!("{name}-{}", ulid::Ulid::new()))
}

/// `Command::args` returns `&mut Command` (for chaining), but `run_logged`
/// needs to own the built command -- this builds one in a single expression
/// without fighting that borrow.
fn shell(program: impl AsRef<std::ffi::OsStr>, args: &[&str]) -> Command {
    let mut c = Command::new(program);
    c.args(args);
    c
}

/// As [`shell`], but running in `dir` (build steps need to run inside the
/// freshly cloned source tree).
fn shell_in(program: impl AsRef<std::ffi::OsStr>, args: &[&str], dir: &Path) -> Command {
    let mut c = shell(program, args);
    c.current_dir(dir);
    c
}

/// Resolve (and, if necessary, provision) whatever the configured backend
/// needs, mutating `cfg` in place with whatever concrete values were
/// resolved (an absolute binary path, a minted token, ...). `data_dir` must
/// already exist.
pub async fn ensure_backend_ready(cfg: &mut BackendConfig, data_dir: &Path) -> Result<()> {
    match cfg {
        BackendConfig::Docker => Ok(()),
        BackendConfig::Smolvm { binary } => ensure_smolvm(binary, data_dir).await,
        BackendConfig::Hypeman { endpoint, token, cli_binary } => {
            ensure_hypeman(endpoint, token, cli_binary, data_dir).await
        }
    }
}

// ---------------------------------------------------------------------------
// SmolVM
// ---------------------------------------------------------------------------

/// Fails fast, with an actionable message, instead of letting a missing
/// `/dev/kvm` surface as a cryptic boot-time error from deep inside a VM
/// fork attempt. Checked before any provisioning work (fetch or build) so a
/// host that can never run SmolVM doesn't first pay for downloading or
/// compiling it.
///
/// Linux-only: on macOS/Windows, libkrun uses Hypervisor.framework/WHP
/// instead, neither of which is a `/dev/kvm`-shaped device file.
#[cfg(target_os = "linux")]
fn check_kvm_available() -> Result<()> {
    use std::fs::OpenOptions;
    match OpenOptions::new().read(true).write(true).open("/dev/kvm") {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!(
            "SmolVM needs /dev/kvm (hardware virtualization) but this host doesn't have it. \
             This is a hardware/platform limit, not something SmolVM or BLAST can work around: \
             on AWS, only `.metal` instance types expose it -- standard Nitro instances (even \
             large, non-metal ones) don't, regardless of CPU count; other clouds have their own \
             \"bare metal\" or \"nested virtualization\" instance flags. Use the docker backend \
             instead, or move this worker to a host/instance type with KVM support."
        ),
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => bail!(
            "SmolVM needs /dev/kvm but got permission denied opening it. If running in a \
             container, it needs /dev/kvm passed through and (typically) to run privileged; \
             outside a container, the process needs to be in the `kvm` group or run as root."
        ),
        Err(e) => Err(e).context("checking /dev/kvm"),
    }
}

#[cfg(not(target_os = "linux"))]
fn check_kvm_available() -> Result<()> {
    Ok(())
}

async fn ensure_smolvm(binary: &mut PathBuf, data_dir: &Path) -> Result<()> {
    check_kvm_available()?;

    if let Ok(resolved) = which::which(&*binary) {
        info!(path = %resolved.display(), "smolvm: found on PATH");
        *binary = resolved;
        return Ok(());
    }

    let managed = managed_bin(data_dir, "smolvm");
    if binary_is_usable(&managed).await {
        info!(path = %managed.display(), "smolvm: using previously auto-provisioned binary");
        *binary = managed;
        return Ok(());
    }

    info!(dest = %managed.display(), "smolvm: not found on PATH or at the managed location");
    match fetch_smolvm_release(&managed, data_dir).await {
        Ok(()) => {
            info!(path = %managed.display(), "smolvm: installed prebuilt release");
            *binary = managed;
            return Ok(());
        }
        Err(e) => {
            warn!(error = %format!("{e:#}"), "smolvm: prebuilt release unavailable, falling back to building from source");
        }
    }

    info!(
        dest = %managed.display(),
        "smolvm: building from source (git clone + cargo build --release, this can take 1-2 minutes)...",
    );
    build_smolvm(&managed, data_dir).await?;
    info!(path = %managed.display(), "smolvm: build complete");
    *binary = managed;
    Ok(())
}

/// Pinned to the exact smolvm release this backend was integration-tested
/// against. A prebuilt tarball trades the source build's implicit
/// "whatever's on main" freshness for a known-good, reproducible artifact --
/// bump deliberately, not automatically, same reasoning as any other vendored
/// dependency pin.
const SMOLVM_RELEASE_VERSION: &str = "1.13.3";

/// Prebuilt release platforms currently published at
/// `github.com/calebwin/smolvm/releases`. `None` here just means "no fast
/// path" -- `ensure_smolvm` falls back to building from source, which covers
/// every platform smolvm itself supports.
fn smolvm_release_platform() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "aarch64") => Some("linux-aarch64"),
        ("linux", "x86_64") => Some("linux-x86_64"),
        _ => None,
    }
}

/// Fetches the pinned prebuilt release tarball and installs it into the same
/// `<data_dir>/bin/{smolvm,lib/}` + `<data_dir>/smolvm/agent-rootfs` layout
/// `build_smolvm` produces, so nothing downstream (`SmolvmBackend`, smolvm's
/// own `find_lib_dir`) needs to know or care which path provisioned it.
/// Seconds instead of minutes, and no Rust/musl toolchain needed on the host
/// -- just network access and `tar`, which every non-exotic host already has.
async fn fetch_smolvm_release(dest: &Path, data_dir: &Path) -> Result<()> {
    let platform = smolvm_release_platform().context("no prebuilt release for this OS/arch")?;
    require_tool("tar", "installing the prebuilt smolvm release")?;

    let url = format!(
        "https://github.com/calebwin/smolvm/releases/download/v{SMOLVM_RELEASE_VERSION}/smolvm-{SMOLVM_RELEASE_VERSION}-{platform}.tar.gz"
    );
    let scratch = build_scratch_dir(data_dir, "smolvm-release");
    tokio::fs::create_dir_all(&scratch).await.context("create scratch dir")?;
    fetch_and_extract_tarball(&url, &scratch).await.context("fetch smolvm release")?;

    install_binary(&scratch.join("smolvm"), dest).await.context("install smolvm binary")?;
    let lib_dest = dest.parent().map_or_else(|| scratch.join("lib"), |p| p.join("lib"));
    install_dir_contents(&scratch.join("lib"), &lib_dest).await.context("install libkrun/libkrunfw")?;

    // Same destination `build_smolvm` uses (see its comment on why
    // XDG_DATA_HOME/data_dir.join("smolvm") is where SmolvmBackend expects it).
    let rootfs_dest = data_dir.join("smolvm").join("agent-rootfs");
    if let Some(parent) = rootfs_dest.parent() {
        tokio::fs::create_dir_all(parent).await.context("create smolvm data dir")?;
    }
    let status = shell("cp", &["-a", &scratch.join("agent-rootfs").to_string_lossy(), &rootfs_dest.to_string_lossy()])
        .status()
        .await
        .context("spawn cp for agent-rootfs")?;
    if !status.success() {
        bail!("installing agent-rootfs failed with {status}");
    }

    let _ = tokio::fs::remove_dir_all(&scratch).await;
    Ok(())
}

async fn build_smolvm(dest: &Path, data_dir: &Path) -> Result<()> {
    // `cargo` is frequently only on PATH via an interactive shell's rcfile
    // (`~/.cargo/env` sourced from `.bashrc`/`.profile`), which a plain
    // `Command::spawn()` from a non-login process -- e.g. blast itself
    // running under systemd, or launched via a bare `nohup` -- never sees.
    // Same class of problem as `go` below; same fix.
    let cargo = find_cargo().context(
        "`cargo` is required to auto-provision smolvm (cargo build --release --bin smolvm) but \
         wasn't found on PATH or at $HOME/.cargo/bin/cargo; install Rust, or point BLAST at an \
         already-running smolvm instance instead",
    )?;
    let cargo_dir = cargo.parent().map_or_else(PathBuf::new, Path::to_path_buf);
    // rustup conventionally lives right next to cargo (`~/.cargo/bin/{cargo,rustup,...}`).
    let rustup = find_rustup().context(
        "`rustup` is required to auto-provision smolvm (its guest agent needs a musl target) \
         but wasn't found on PATH or at $HOME/.cargo/bin/rustup; install Rust via rustup, or \
         point BLAST at an already-running smolvm instance instead",
    )?;
    require_tool("git", "git clone smolvm")?;
    require_tool("bash", "smolvm's agent-rootfs build script")?;
    // smolvm vendors libkrun/libkrunfw (the actual hypervisor libraries, not
    // something `cargo build` produces) as Git LFS objects. A plain `git
    // clone` only materializes their real bytes -- instead of ~130-byte text
    // pointer files that *look* like valid installs right up until a VM
    // actually tries to boot -- if the LFS smudge/clean filter is already
    // registered in git config. That's true on a host someone has already
    // run `git lfs install` on, which is not guaranteed on a fresh one, so
    // do it ourselves (idempotent; safe to run unconditionally).
    ensure_git_lfs().await?;

    let src = build_scratch_dir(data_dir, "smolvm");
    run_logged(
        shell(
            "git",
            &["clone", "--depth", "1", "https://github.com/calebwin/smolvm", &src.to_string_lossy()],
        ),
        "smolvm: git clone",
    )
    .await?;
    // Belt-and-suspenders: confirms (and, if the initial checkout's smudge
    // pass somehow missed anything, repairs) that the LFS-tracked files are
    // real content rather than pointers, without having to duplicate
    // `find_lib_dir`'s exact file list here to check it ourselves.
    run_logged(shell_in("git-lfs", &["pull"], &src), "smolvm: git lfs pull (libkrun/libkrunfw)").await?;

    run_logged(
        shell_in(&cargo, &["build", "--release", "--bin", "smolvm"], &src),
        "smolvm: cargo build --release",
    )
    .await?;

    // The CLI binary alone can't actually boot a VM -- confirmed the hard way
    // by watching `machine create`+`start` fail on a from-scratch build even
    // though `smolvm --version` succeeds fine. Two more things are needed:
    //
    //   1. A separately-built "agent-rootfs" (a small Alpine image plus a
    //      musl-compiled `smolvm-agent` guest binary), which smolvm looks for
    //      at a fixed, standard location -- `~/.local/share/smolvm/agent-rootfs`
    //      -- regardless of where the `smolvm` binary itself lives. It's built
    //      by the repo's own `scripts/build-agent-rootfs.sh --install`, which
    //      needs the musl target added to the *pinned* toolchain first (the
    //      repo carries a `rust-toolchain.toml` override, so `rustup target
    //      add` has to run with the clone as the working directory for it to
    //      land on the right toolchain).
    //   2. libkrun/libkrunfw: prebuilt `.so` files vendored straight into the
    //      repo at `lib/linux-<arch>/` (not something `cargo build` produces).
    //      smolvm `dlopen()`s these at VM-boot time, searching `<exe_dir>/lib/`
    //      among other places (see smolvm's own `find_lib_dir`) -- so once we
    //      move the binary to our managed location, its libs need to move
    //      there with it, in a `lib/` subdirectory right beside it.
    let musl_target = musl_target_triple();
    run_logged(
        shell_in(&rustup, &["target", "add", musl_target], &src),
        "smolvm: rustup target add (guest agent)",
    )
    .await?;

    let mut rootfs_cmd = shell("bash", &["scripts/build-agent-rootfs.sh", "--install"]);
    rootfs_cmd.current_dir(&src);
    // The script shells out to bare `cargo`/`rustup` itself to cross-compile
    // the guest agent; give it the same PATH fix we needed one level up, or
    // it hits the identical "not found" problem all over again.
    prepend_path(&mut rootfs_cmd, &cargo_dir);
    // Without this, --install writes to smolvm's own fixed default
    // (~/.local/share/smolvm/agent-rootfs, see the comment above) regardless
    // of our data_dir. build-agent-rootfs.sh reads $XDG_DATA_HOME directly
    // (DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/smolvm") rather than
    // going through smolvm's own SMOLVM_DATA_DIR relocation (that's binary-side
    // logic a bash script doesn't run), so XDG_DATA_HOME is the one that
    // actually lands it where SmolvmBackend::new (SMOLVM_DATA_DIR-based) will
    // later look: data_dir.join("smolvm").
    rootfs_cmd.env("XDG_DATA_HOME", data_dir);
    run_logged(rootfs_cmd, "smolvm: build + install agent-rootfs").await?;

    install_binary(&src.join("target/release/smolvm"), dest).await?;
    let lib_src = src.join("lib").join(format!("linux-{}", std::env::consts::ARCH));
    let lib_dest = dest.parent().map_or_else(|| lib_src.clone(), |p| p.join("lib"));
    install_dir_contents(&lib_src, &lib_dest).await.context("install libkrun/libkrunfw")?;

    let _ = tokio::fs::remove_dir_all(&src).await;
    Ok(())
}

fn musl_target_triple() -> &'static str {
    if std::env::consts::ARCH == "aarch64" {
        "aarch64-unknown-linux-musl"
    } else {
        "x86_64-unknown-linux-musl"
    }
}

/// Unlike `ensure_erofs_utils` below, this one is fatal if it can't succeed:
/// without it, smolvm silently "installs" with ~130-byte pointer files in
/// place of libkrun/libkrunfw and fails much later, cryptically, the first
/// time something actually tries to boot a VM -- worse than just erroring
/// now, while we still know exactly what's missing and why.
async fn ensure_git_lfs() -> Result<()> {
    if which::which("git-lfs").is_err() {
        info!("smolvm: git-lfs not found; installing (needed for smolvm's vendored libkrun/libkrunfw)...");
        let installed = passwordless_sudo().await
            && Command::new("sudo")
                .args(["-n", "apt-get", "install", "-y", "git-lfs"])
                .status()
                .await
                .is_ok_and(|s| s.success());
        if !installed {
            bail!(
                "`git-lfs` is required to fetch smolvm's vendored libkrun/libkrunfw libraries but \
                 isn't installed and couldn't be installed automatically (no passwordless sudo, or \
                 not a Debian/Ubuntu host); install it manually (apt-get install git-lfs) and retry"
            );
        }
    }
    // Registers the smudge/clean filters in git's (global) config; idempotent,
    // so unconditionally re-running it here is cheap and safe regardless of
    // whether some earlier process on this host already did.
    let status = Command::new("git-lfs").arg("install").status().await.context("git lfs install")?;
    if !status.success() {
        bail!("git lfs install failed with {status}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Hypeman
// ---------------------------------------------------------------------------

async fn ensure_hypeman(
    endpoint: &str,
    token: &mut String,
    cli_binary: &mut PathBuf,
    data_dir: &Path,
) -> Result<()> {
    if endpoint_reachable(endpoint).await {
        info!(endpoint = %endpoint, "hypeman: endpoint already reachable, using it as configured");
        if token.is_empty() {
            warn!(
                endpoint = %endpoint,
                "hypeman: endpoint is reachable but no token is configured \
                 (BLAST__BACKEND__TOKEN / backend.token); every request will be \
                 rejected as unauthorized until one is set",
            );
        }
        resolve_hypeman_cli(cli_binary, data_dir).await;
        return Ok(());
    }

    info!(endpoint = %endpoint, "hypeman: endpoint not reachable, auto-provisioning a local instance");

    let server_bin = managed_bin(data_dir, "hypeman");
    let cli_bin = managed_bin(data_dir, "hypeman-cli");
    let genjwt_bin = managed_bin(data_dir, "gen-jwt");

    let have_all = binary_is_usable(&server_bin).await
        && binary_is_usable(&cli_bin).await
        && binary_is_usable(&genjwt_bin).await;
    if have_all {
        info!("hypeman: using previously auto-provisioned binaries");
    } else {
        // Upstream's prebuilt hypeman-api release (github.com/kernel/hypeman)
        // was tried here and reverted: instance boots reliably hang in
        // "Initializing" against it (isolated head-to-head against a
        // source-built binary of the identical ./cmd/api package, same host,
        // same freshly-fetched kernel/system-files version -- only the
        // binary provenance differed). A fast provisioning path that produces
        // a broken server is worse than a slow one that works, so this stays
        // on the build-from-source path until that's root-caused upstream.
        build_hypeman(&server_bin, &cli_bin, &genjwt_bin, data_dir).await?;
        info!("hypeman: build complete");
    }

    ensure_erofs_utils().await;

    let secret = random_secret().await.context("generate hypeman JWT secret")?;
    let minted = mint_jwt(&genjwt_bin, &secret).await.context("mint hypeman JWT")?;

    let hm_data_dir = data_dir.join("hypeman");
    tokio::fs::create_dir_all(&hm_data_dir).await.context("create hypeman data dir")?;
    let port = port_from_endpoint(endpoint).unwrap_or(4973);
    let config_path = hm_data_dir.join("config.yaml");
    tokio::fs::write(
        &config_path,
        format!(
            "# auto-generated by blast's backend bootstrap.\n\
             # jwt_secret here is cosmetic: hypeman only honors JWT_SECRET as an env var.\n\
             jwt_secret: \"{secret}\"\n\
             data_dir: {}\n\
             port: {port}\n",
            hm_data_dir.display(),
        ),
    )
    .await
    .context("write hypeman config")?;

    spawn_hypeman_server(&server_bin, &secret, &config_path, &hm_data_dir).await?;

    wait_for_endpoint(endpoint, Duration::from_secs(30))
        .await
        .context("hypeman server did not become healthy in time")?;

    *token = minted;
    *cli_binary = cli_bin;
    info!(endpoint = %endpoint, "hypeman: auto-provisioned server is up");
    Ok(())
}

async fn resolve_hypeman_cli(cli_binary: &mut PathBuf, data_dir: &Path) {
    if which::which(&*cli_binary).is_ok() {
        return;
    }
    let managed = managed_bin(data_dir, "hypeman-cli");
    if binary_is_usable(&managed).await {
        *cli_binary = managed;
    }
}

/// Downloads `url` and extracts it into `dest_dir` (already created). Used by
/// `fetch_smolvm_release`; factored out so a future prebuilt-release fast
/// path doesn't have to duplicate this.
async fn fetch_and_extract_tarball(url: &str, dest_dir: &Path) -> Result<()> {
    info!(url, "fetching prebuilt release");
    let resp = reqwest::get(url).await.context("download release")?;
    if !resp.status().is_success() {
        bail!("release download failed: HTTP {}", resp.status());
    }
    let bytes = resp.bytes().await.context("read release body")?;
    let tarball = dest_dir.join("release.tar.gz");
    tokio::fs::write(&tarball, &bytes).await.context("write release tarball")?;
    let status = shell_in("tar", &["xzf", "release.tar.gz"], dest_dir).status().await.context("spawn tar")?;
    if !status.success() {
        bail!("tar extraction failed with {status}");
    }
    let _ = tokio::fs::remove_file(&tarball).await;
    Ok(())
}

async fn build_hypeman(
    server_bin: &Path,
    cli_bin: &Path,
    genjwt_bin: &Path,
    data_dir: &Path,
) -> Result<()> {
    let go = find_tool("go", &["/usr/local/go/bin/go"]).context(
        "`go` is required to auto-provision hypeman but wasn't found on PATH or at \
         /usr/local/go/bin/go; install Go, or point BLAST at an already-running hypeman \
         instance via backend.endpoint instead",
    )?;
    let go_dir = go.parent().map_or_else(PathBuf::new, Path::to_path_buf);
    // hypeman's real build is `make build-linux`, not a bare `go build`: that
    // target chains `ensure-ch-binaries` (downloads Cloud Hypervisor),
    // `ensure-firecracker-binaries` (downloads Firecracker), `ensure-caddy-binaries`
    // (builds Caddy w/ its DNS module via xcaddy), and `build-embedded` (cross-compiles
    // the in-VM guest-agent and init binaries) *before* the actual
    // `go build -tags containers_image_openpgp -o bin/hypeman ./cmd/api`. A bare
    // `go build ./cmd/api` skips all of that -- confirmed the hard way: it compiles
    // (Go doesn't know it's missing embedded assets) but the resulting binary can't
    // actually run a microVM. Shelling out to hypeman's own `make build-linux`
    // instead of replicating its exact steps here means this keeps matching
    // upstream as its Makefile evolves (new CH/Firecracker versions, etc.)
    // without BLAST needing a matching update every time.
    let make = find_tool("make", &["/usr/bin/make"]).context(
        "`make` is required to auto-provision hypeman (its real build is `make \
         build-linux`, which fetches/builds the embedded Cloud Hypervisor, \
         Firecracker, and Caddy binaries a bare `go build` would silently skip) but \
         wasn't found on PATH; install build-essential (or equivalent), or point \
         BLAST at an already-running hypeman instance via backend.endpoint instead",
    )?;
    require_tool("git", "git clone hypeman")?;

    info!(
        "hypeman: building server (make build-linux -- fetches/builds embedded Cloud \
         Hypervisor, Firecracker, and Caddy binaries plus the guest-agent/init, this \
         can take several minutes on a cold cache)...",
    );
    let server_src = build_scratch_dir(data_dir, "hypeman");
    run_logged(
        shell(
            "git",
            &["clone", "--depth", "1", "https://github.com/kernel/hypeman", &server_src.to_string_lossy()],
        ),
        "hypeman: git clone (server)",
    )
    .await?;
    // `make` itself shells out to a bare `go` internally, hitting the exact same
    // PATH gap `find_tool` above just worked around for us -- hand it down.
    let mut make_cmd = shell_in(&make, &["build-linux"], &server_src);
    prepend_path(&mut make_cmd, &go_dir);
    run_logged(make_cmd, "hypeman: make build-linux").await?;
    // `make build-linux` doesn't produce gen-jwt (upstream's Makefile only offers
    // an interactive `go run`-based `make gen-jwt`); build it ourselves the same
    // way the Makefile builds any other plain, no-embedded-assets cmd/ binary.
    run_logged(
        shell_in(&go, &["build", "-o", "bin/gen-jwt", "./cmd/gen-jwt"], &server_src),
        "hypeman: go build ./cmd/gen-jwt",
    )
    .await?;
    install_binary(&server_src.join("bin/hypeman"), server_bin).await?;
    install_binary(&server_src.join("bin/gen-jwt"), genjwt_bin).await?;
    let _ = tokio::fs::remove_dir_all(&server_src).await;

    info!("hypeman: building CLI (go build)...");
    let cli_src = build_scratch_dir(data_dir, "hypeman-cli");
    run_logged(
        shell(
            "git",
            &["clone", "--depth", "1", "https://github.com/kernel/hypeman-cli", &cli_src.to_string_lossy()],
        ),
        "hypeman: git clone (cli)",
    )
    .await?;
    run_logged(
        shell_in(&go, &["build", "-o", "hypeman", "./cmd/hypeman"], &cli_src),
        "hypeman: go build ./cmd/hypeman",
    )
    .await?;
    install_binary(&cli_src.join("hypeman"), cli_bin).await?;
    let _ = tokio::fs::remove_dir_all(&cli_src).await;

    Ok(())
}

/// Best-effort: hypeman needs `mkfs.erofs` (package `erofs-utils`) on the host
/// to convert OCI images. Not fatal if we can't install it -- image-backed
/// forks will just fail later with a clearer error from hypeman itself.
async fn ensure_erofs_utils() {
    if which::which("mkfs.erofs").is_ok() {
        return;
    }
    info!("hypeman: mkfs.erofs not found; installing erofs-utils (needed for OCI image conversion)...");
    let installed = passwordless_sudo().await
        && Command::new("sudo")
            .args(["-n", "apt-get", "install", "-y", "erofs-utils"])
            .status()
            .await
            .is_ok_and(|s| s.success());
    if !installed {
        warn!(
            "hypeman: could not install erofs-utils automatically (no passwordless sudo, or \
             not a Debian/Ubuntu host); hypeman may fail to convert OCI images until \
             `erofs-utils` is installed manually",
        );
    }
}

/// Random 32-byte hex secret for hypeman's `JWT_SECRET`, read straight from
/// the kernel CSPRNG. Deliberately not the `rand` crate: `/dev/urandom` needs
/// no new dependency and is exactly as good a source on the Linux hosts BLAST
/// runs on (it already assumes Linux for the microVM backends).
async fn random_secret() -> Result<String> {
    let mut buf = [0u8; 32];
    let mut f = tokio::fs::File::open("/dev/urandom").await.context("open /dev/urandom")?;
    f.read_exact(&mut buf).await.context("read /dev/urandom")?;
    Ok(hex::encode(buf))
}

async fn mint_jwt(genjwt_bin: &Path, secret: &str) -> Result<String> {
    let out = Command::new(genjwt_bin)
        .env("JWT_SECRET", secret)
        .args(["-user-id", "blast", "-duration", "8760h"])
        .output()
        .await
        .context("run gen-jwt")?;
    if !out.status.success() {
        bail!("gen-jwt failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    String::from_utf8(out.stdout).context("gen-jwt output not utf8").map(|s| s.trim().to_owned())
}

fn port_from_endpoint(endpoint: &str) -> Option<u16> {
    reqwest::Url::parse(endpoint).ok().and_then(|u| u.port())
}

async fn endpoint_reachable(endpoint: &str) -> bool {
    let Ok(client) = reqwest::Client::builder().timeout(Duration::from_secs(2)).build() else {
        return false;
    };
    let url = format!("{}/health", endpoint.trim_end_matches('/'));
    client.get(url).send().await.is_ok_and(|r| r.status().is_success())
}

async fn wait_for_endpoint(endpoint: &str, timeout: Duration) -> Result<()> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if endpoint_reachable(endpoint).await {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            bail!("{endpoint}/health never returned healthy");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

// ---------------------------------------------------------------------------
// CAP_NET_ADMIN for the spawned hypeman server
// ---------------------------------------------------------------------------

/// Strategy for launching a process that needs `CAP_NET_ADMIN` (hypeman's
/// network bridge setup).
///
/// Important nuance, worth spelling out because it's easy to get backwards:
/// granting `CAP_NET_ADMIN` to *blast's own* binary (e.g. `setcap
/// cap_net_admin+ep $(which blast)`) would do **nothing** for a spawned
/// hypeman child. Linux file capabilities are evaluated by the kernel at
/// `execve()` time against the file being executed, not inherited from the
/// parent's effective set; carrying a capability across `fork`+`exec` into an
/// unrelated child requires the *ambient* capability set, which an ordinary
/// `Command::spawn()` never populates (and isn't worth wiring up here just
/// for one backend). Concretely: the *child* needs its own elevation, not
/// the parent. That leaves three things that actually work, tried here
/// cheapest/most-persistent first:
///   1. `blast` is already root -- a root child is root too, no capability
///      plumbing needed.
///   2. The hypeman binary *itself* already carries the file capability
///      (`getcap` reports `cap_net_admin`) -- it runs privileged on its own
///      regardless of blast's privilege.
///   3. Passwordless sudo works -- either stamp the capability onto the
///      binary once (persists across restarts, no more sudo needed after
///      that) or, if that fails (e.g. a `nosuid`-mounted `data_dir` strips
///      file capabilities), fall back to wrapping every launch in `sudo`.
enum CapStrategy {
    /// Already privileged enough -- spawn the binary directly.
    Direct,
    /// Not privileged, but passwordless sudo is available -- wrap the spawn.
    Sudo,
    /// Nothing available; caller should warn and give up.
    Unavailable,
}

async fn cap_net_admin_strategy(target: &Path) -> CapStrategy {
    if binary_has_cap_net_admin(target).await || running_as_root().await {
        return CapStrategy::Direct;
    }
    if !passwordless_sudo().await {
        return CapStrategy::Unavailable;
    }
    // One-time elevation, best-effort: try to stamp the capability onto the
    // binary file itself so future restarts don't need sudo at all.
    let _ = Command::new("sudo").args(["-n", "setcap", "cap_net_admin+ep"]).arg(target).status().await;
    if binary_has_cap_net_admin(target).await {
        CapStrategy::Direct
    } else {
        CapStrategy::Sudo
    }
}

async fn binary_has_cap_net_admin(path: &Path) -> bool {
    Command::new("getcap")
        .arg(path)
        .output()
        .await
        .ok()
        .filter(|o| o.status.success())
        .is_some_and(|o| String::from_utf8_lossy(&o.stdout).contains("cap_net_admin"))
}

async fn running_as_root() -> bool {
    Command::new("id")
        .arg("-u")
        .output()
        .await
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .is_some_and(|s| s.trim() == "0")
}

async fn passwordless_sudo() -> bool {
    Command::new("sudo").args(["-n", "true"]).status().await.is_ok_and(|s| s.success())
}

async fn spawn_hypeman_server(
    server_bin: &Path,
    secret: &str,
    config_path: &Path,
    hm_data_dir: &Path,
) -> Result<()> {
    let strategy = cap_net_admin_strategy(server_bin).await;
    if matches!(strategy, CapStrategy::Unavailable) {
        warn!(
            binary = %server_bin.display(),
            "hypeman needs CAP_NET_ADMIN for its network bridge setup, and neither root \
             privileges nor passwordless sudo are available to blast right now. Setting a \
             capability on blast's own binary would NOT fix this -- capabilities don't cross \
             exec into a child that lacks them itself. Fix one of: run blast as root; run \
             `sudo setcap cap_net_admin+ep {}` yourself (persists across restarts); or \
             configure passwordless sudo for this user. Falling back to a backend that isn't \
             configured will fail until then.",
            server_bin.display(),
        );
        bail!("cannot launch hypeman: CAP_NET_ADMIN unavailable (see warning above)");
    }

    let log_path = hm_data_dir.join("hypeman.log");
    let log_out = std::fs::File::create(&log_path).context("create hypeman log file")?;
    let log_err = log_out.try_clone().context("dup hypeman log file")?;

    // hypeman has no --config flag: cmd/api/main.go reads only the
    // CONFIG_PATH env var (`os.Getenv("CONFIG_PATH")`), so an argv flag is
    // silently ignored (nothing in main() parses os.Args) and it falls back
    // to its compiled-in default data_dir of /var/lib/hypeman -- which is
    // how it looked "zero-config" while actually ignoring our config file
    // entirely for everything except JWT_SECRET, which we already pass
    // as an env var.
    let config_path_str = config_path.to_string_lossy().into_owned();
    let mut cmd = match strategy {
        CapStrategy::Direct => {
            let mut c = Command::new(server_bin);
            c.env("JWT_SECRET", secret);
            c.env("CONFIG_PATH", &config_path_str);
            c
        }
        CapStrategy::Sudo => {
            let mut c = Command::new("sudo");
            c.args([
                "-n",
                "env",
                &format!("JWT_SECRET={secret}"),
                &format!("CONFIG_PATH={config_path_str}"),
            ])
            .arg(server_bin);
            c
        }
        CapStrategy::Unavailable => unreachable!("handled above"),
    };
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::from(log_out));
    cmd.stderr(Stdio::from(log_err));
    // New process group: a Ctrl-C to blast's own foreground terminal (SIGINT
    // to the whole process group) shouldn't take hypeman down with it. It's
    // meant to outlive this particular blast invocation.
    cmd.process_group(0);

    cmd.spawn().context("spawn hypeman server")?;
    // Deliberately not awaited: this is a detached background server for
    // BLAST's whole lifetime, not a subprocess whose exit we need to observe
    // here. Tokio's runtime still reaps it in the background (no zombie) as
    // long as the runtime is alive, which for `blast` itself is forever.
    info!(log = %log_path.display(), "hypeman: server spawned");
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared build/tool helpers
// ---------------------------------------------------------------------------

async fn binary_is_usable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    tokio::time::timeout(Duration::from_secs(5), Command::new(path).arg("--version").output())
        .await
        .ok()
        .and_then(std::result::Result::ok)
        .is_some()
}

fn find_tool(name: &str, extra_candidates: &[&str]) -> Option<PathBuf> {
    if let Ok(p) = which::which(name) {
        return Some(p);
    }
    extra_candidates.iter().map(PathBuf::from).find(|p| p.is_file())
}

/// `rustup`'s installer puts `cargo` at `$HOME/.cargo/bin/cargo` and relies on
/// an rcfile (`~/.cargo/env`, sourced from `.bashrc`/`.profile`) to put it on
/// `PATH` -- which only happens for interactive/login shells, not a plain
/// `Command::spawn()` from a non-login process. `$HOME` is a plain env var
/// rather than a `&'static str`, so this can't reuse `find_tool` directly.
fn find_cargo() -> Option<PathBuf> {
    if let Ok(p) = which::which("cargo") {
        return Some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let candidate = PathBuf::from(home).join(".cargo/bin/cargo");
    candidate.is_file().then_some(candidate)
}

/// As [`find_cargo`], for `rustup` (same installer, same directory, same PATH problem).
fn find_rustup() -> Option<PathBuf> {
    if let Ok(p) = which::which("rustup") {
        return Some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let candidate = PathBuf::from(home).join(".cargo/bin/rustup");
    candidate.is_file().then_some(candidate)
}

/// Prepend `dir` to a child command's `PATH`. Needed when the child is itself
/// a script that shells out to a bare tool name (e.g. smolvm's
/// `build-agent-rootfs.sh` calling plain `cargo`) -- our own process's `PATH`
/// may already be missing that tool (see `find_cargo`/`find_rustup`), and a
/// spawned script inherits that same gap one level down unless we hand it a
/// corrected `PATH` explicitly.
fn prepend_path(cmd: &mut Command, dir: &Path) {
    let existing = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![dir.to_path_buf()];
    paths.extend(std::env::split_paths(&existing));
    if let Ok(joined) = std::env::join_paths(paths) {
        cmd.env("PATH", joined);
    }
}

fn require_tool(name: &str, needed_for: &str) -> Result<()> {
    if which::which(name).is_ok() {
        return Ok(());
    }
    bail!(
        "`{name}` is required to auto-provision this backend ({needed_for}) but isn't on \
         PATH; install it, or point BLAST at an already-running instance instead"
    );
}

async fn install_binary(built: &Path, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await.context("create managed bin dir")?;
    }
    tokio::fs::copy(built, dest)
        .await
        .with_context(|| format!("install {} -> {}", built.display(), dest.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = tokio::fs::metadata(dest).await?.permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(dest, perms).await?;
    }
    Ok(())
}

/// Flat, non-recursive copy of every file directly inside `src_dir` into
/// `dest_dir` (used for smolvm's vendored `lib/linux-<arch>/*.so*`). Symlinks
/// are dereferenced rather than preserved -- harmless here, since a `dlopen()`
/// by exact filename only needs the destination path to exist and contain
/// valid library bytes, not to itself be a symlink.
async fn install_dir_contents(src_dir: &Path, dest_dir: &Path) -> Result<()> {
    tokio::fs::create_dir_all(dest_dir).await.context("create dest dir")?;
    let mut entries = tokio::fs::read_dir(src_dir).await.context("read src dir")?;
    while let Some(entry) = entries.next_entry().await.context("read src dir entry")? {
        let dest = dest_dir.join(entry.file_name());
        tokio::fs::copy(entry.path(), &dest)
            .await
            .with_context(|| format!("copy {}", entry.path().display()))?;
    }
    Ok(())
}

/// Run a build step to completion, streaming its output through `tracing`
/// (at debug, since a full `cargo build`/`go build` is thousands of lines)
/// plus a periodic info-level heartbeat so a slow build doesn't look hung.
async fn run_logged(mut cmd: Command, ctx: &str) -> Result<()> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().with_context(|| format!("spawn: {ctx}"))?;
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let out_task = tokio::spawn(drain_lines(stdout, ctx.to_owned()));
    let err_task = tokio::spawn(drain_lines(stderr, ctx.to_owned()));

    let start = std::time::Instant::now();
    let status = loop {
        tokio::select! {
            status = child.wait() => break status.with_context(|| format!("wait: {ctx}"))?,
            () = tokio::time::sleep(Duration::from_secs(15)) => {
                info!("{ctx}: still running ({}s elapsed)...", start.elapsed().as_secs());
            }
        }
    };
    let _ = out_task.await;
    let _ = err_task.await;
    if !status.success() {
        bail!("{ctx} failed: {status}");
    }
    Ok(())
}

async fn drain_lines(io: impl AsyncRead + Unpin, ctx: String) {
    let mut lines = BufReader::new(io).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        debug!("{ctx}: {line}");
    }
}
