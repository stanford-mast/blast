#!/usr/bin/env python3
import asyncio
import os
import re
import shlex
import socket
import subprocess
import sys
import platform
import tempfile
from pathlib import Path
from shutil import which
from playwright.async_api import async_playwright

# ─── Helper: detect OS and install missing dependencies ───────────────────────

def install_with_apt(packages):
    cmd = ["sudo", "apt-get", "update"]
    subprocess.run(cmd, check=False)
    cmd = ["sudo", "apt-get", "install", "-y"] + packages
    subprocess.run(cmd, check=True)

def install_with_yum(packages):
    cmd = ["sudo", "yum", "install", "-y"] + packages
    subprocess.run(cmd, check=True)

def install_with_brew(packages):
    cmd = ["brew", "install"] + packages
    subprocess.run(cmd, check=True)

def ensure_noVNC():
    noVNC_dir = Path.home() / "noVNC"
    if not noVNC_dir.exists():
        print("Cloning noVNC into ~/noVNC …")
        subprocess.run(
            ["git", "clone", "https://github.com/novnc/noVNC.git", str(noVNC_dir)],
            check=True
        )
    # Ensure novnc_proxy is executable
    proxy = noVNC_dir / "utils" / "novnc_proxy"
    if not proxy.exists():
        print("❌ noVNC proxy script not found after clone.")
        sys.exit(1)

def check_and_install_dependencies():
    system = platform.system()

    # Common dependencies: tigervnc (Xvnc), fluxbox, git, python3-pip (for Playwright)
    missing = []

    # 1) Check for Xvnc (TigerVNC)
    if which("Xvnc") is None:
        missing.append("tigervnc-standalone-server")

    # 2) Check for fluxbox
    if which("fluxbox") is None:
        missing.append("fluxbox")

    # 3) Check for git (needed for noVNC)
    if which("git") is None:
        missing.append("git")

    if which("matchbox-window-manager") is None:
        missing.append("matchbox-window-manager")

    # 4) Python3 and pip should already be present if script is running
    #    But ensure Playwright is installed:
    try:
        import playwright  # noqa
    except ImportError:
        missing.append("python3-pip")

    # 5) Check package manager and install
    if missing:
        print("Missing packages detected:", missing)
        if system == "Linux":
            if which("apt-get"):
                install_with_apt(missing)
            elif which("yum"):
                install_with_yum(missing)
            else:
                print("❌ No apt-get or yum found. Please install:", missing)
                sys.exit(1)
        elif system == "Darwin":
            # On macOS, use Homebrew. Map package names:
            brew_map = {
                "tigervnc-standalone-server": "tigervnc",
                "fluxbox": "fluxbox",
                "git": "git",
                "python3-pip": "python3",
            }
            to_install = [brew_map[p] for p in missing if p in brew_map]
            if len(to_install) != len(missing):
                print("❌ Some dependencies not mapped for brew:", missing)
                sys.exit(1)
            install_with_brew(to_install)
        else:
            print(f"❌ Unsupported OS for automatic install: {system}")
            sys.exit(1)

    # 6) Ensure Playwright is installed and install browsers if needed
    try:
        import playwright  # noqa
    except ImportError:
        print("Installing Playwright for Python…")
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
    # Install browser binaries
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)

    # 7) Ensure noVNC is present
    ensure_noVNC()

# ─── Helper: find a free display & start Xvnc ─────────────────────────────────

async def find_free_display():
    """
    Iterate through display numbers 1..99. For each:
      1. Kill any existing Xvnc (vncserver) on :N.
      2. Remove stale ~/.vnc/X<N>.* files.
      3. Attempt to start Xvnc :N. If it stays alive for ~1s, return (N, proc).
    """
    home = Path(os.environ["HOME"])
    vnc_dir = home / ".vnc"
    vnc_dir.mkdir(exist_ok=True)

    for n in range(1, 100):
        # 1) Kill any existing Xvnc on :N
        subprocess.run(
            ["vncserver", "-kill", f":{n}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # 2) Remove leftover ~/.vnc/X<N>.sock, ~/.vnc/*.pid, ~/.vnc/*.log
        for suffix in [f"X{n}.sock", f"{home.name}:{n}.pid", f"{home.name}:{n}.log"]:
            try:
                (vnc_dir / suffix).unlink(missing_ok=True)
            except PermissionError:
                pass

        # 3) Start Xvnc :N
        try:
            xvnc_proc = await start_xvnc(n)
            return n, xvnc_proc
        except Exception as e:
            print(f"[Display :{n}] Xvnc failed: {e}")
            continue

    raise RuntimeError("No free display found")


async def start_xvnc(display: int):
    """
    Launch Xvnc (TigerVNC) on display :N with no password.
    Poll up to 1s for immediate failure. Returns proc on success.
    """
    home = Path(os.environ["HOME"])
    vnc_dir = home / ".vnc"
    vnc_dir.mkdir(exist_ok=True)

    cmd = [
        "Xvnc",
        f":{display}",
        "-geometry", "1280x720",
        "-depth", "24",
        "-SecurityTypes", "None"
    ]
    print(f"[Xvnc] {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(*cmd)

    # Poll for up to 1s
    for _ in range(10):
        await asyncio.sleep(0.1)
        if proc.returncode not in (None, 0):
            raise RuntimeError(f"Xvnc exited with code {proc.returncode} on :{display}")
    return proc

# ─── Helper: start fluxbox ────────────────────────────────────────────────────

async def start_window_manager(display: int):
    """
    Start a window manager on DISPLAY=:<display> that drops all decorations.
    ─ On Linux: use `matchbox-window-manager -use_titlebar no` so no title bar or buttons appear.
    ─ On macOS: fall back to fluxbox (XQuartz's quartz-wm will still show a titlebar, since matchbox isn't available).
    Returns the subprocess.Process for the chosen WM.
    """
    system = platform.system()
    env = os.environ.copy()
    env["DISPLAY"] = f":{display}"

    if system == "Linux":
        # Use matchbox-window-manager to remove all decorations
        wm_cmd = ["matchbox-window-manager", "-use_titlebar", "no"]
        print(f"[WM] Running (Linux): {' '.join(wm_cmd)}")
        proc = await asyncio.create_subprocess_exec(*wm_cmd, env=env)
        await asyncio.sleep(0.5)
        return proc

    elif system == "Darwin":
        # On macOS, matchbox isn't available. Fall back to fluxbox (QuartzWM is already running under XQuartz).
        wm_cmd = ["fluxbox", "-display", f":{display}"]
        print(f"[WM] Running (macOS fallback): {' '.join(wm_cmd)}")
        proc = await asyncio.create_subprocess_exec(*wm_cmd, env=env)
        await asyncio.sleep(0.5)
        return proc

    else:
        # If some other OS, just attempt fluxbox
        wm_cmd = ["fluxbox", "-display", f":{display}"]
        print(f"[WM] Running (fallback): {' '.join(wm_cmd)}")
        proc = await asyncio.create_subprocess_exec(*wm_cmd, env=env)
        await asyncio.sleep(0.5)
        return proc


# ─── Helper: find free HTTP port & start noVNC ─────────────────────────────────

async def find_free_http_port(start: int = 6080, end: int = 6099):
    """
    Return first free TCP port in [start..end]. If in use by novnc_proxy,
    kill that process and re-check.
    """
    for port in range(start, end + 1):
        with socket.socket() as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                # Kill any novnc_proxy listening on that port
                subprocess.run(
                    ["pkill", "-f", f"novnc_proxy.*--listen {port}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Re-check
                try:
                    sock.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
    raise RuntimeError("No free HTTP port between 6080 and 6099")


async def patch_novnc_ui():
    """Patch noVNC's HTML to hide UI elements and auto-connect"""
    noVNC_dir = Path.home() / "noVNC"
    html_path = noVNC_dir / "vnc.html"
    patch_version = "v0.1.22"

    if not html_path.exists():
        print("\u26a0\ufe0f noVNC HTML not found!")
        return

    html = html_path.read_text()

    # Remove any existing patches
    while True:
        new_html = re.sub(
            r"<!-- custom patch v0\.\d+\.\d+ -->\s*(<style>.*?</style>\s*)?(<script>.*?</script>\s*)?",
            "",
            html,
            flags=re.DOTALL
        )
        if new_html == html:
            break
        html = new_html

    if f"<!-- custom patch {patch_version} -->" in html:
        print(f"\u2705 noVNC UI already patched with {patch_version}.")
        return

    # Create new patch with both CSS and JS
    patch = f"""<!-- custom patch {patch_version} -->
<style>
    #noVNC_control_bar_anchor,
    #noVNC_control_bar,
    #noVNC_status,
    #noVNC_connect_dlg,
    #noVNC_control_bar_hint,
    #noVNC_transition,
    #noVNC_bell,
    #noVNC_fallback_error,
    #noVNC_hint_anchor,
    #noVNC_center {{
        display: none !important;
    }}
</style>
<script>
window.addEventListener('load', function () {{
    const style = document.createElement('style');
    style.textContent = `
        #noVNC_control_bar_anchor,
        #noVNC_control_bar,
        #noVNC_status,
        #noVNC_connect_dlg,
        #noVNC_control_bar_hint,
        #noVNC_transition,
        #noVNC_bell,
        #noVNC_fallback_error,
        #noVNC_hint_anchor,
        #noVNC_center {{
            display: none !important;
        }}
    `;
    document.head.appendChild(style);
    const button = document.querySelector("#noVNC_connect_button");
    if (button) button.click();
}});
</script>"""

    patched = html.replace("</head>", patch + "\n</head>")
    html_path.write_text(patched)

    print(f"\u2705 Patched {html_path} with {patch_version}")

async def start_novnc(display: int, initial_port: int = 6080):
    """
    Launch noVNC's novnc_proxy, forwarding VNC (5900+display) → HTTP.
    Returns (proc, http_port).
    """
    vnc_port = 5900 + display
    noVNC_dir = Path.home() / "noVNC"
    proxy = noVNC_dir / "utils" / "novnc_proxy"

    if not proxy.exists():
        raise FileNotFoundError(f"novnc_proxy not found at {proxy}")

    # Patch noVNC UI before starting proxy
    await patch_novnc_ui()

    port = await find_free_http_port(initial_port, initial_port + 19)
    cmd = ["bash", str(proxy), "--vnc", f"localhost:{vnc_port}", "--web", str(noVNC_dir), "--listen", str(port)]
    print(f"[noVNC] {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(*cmd)
    await asyncio.sleep(0.5)
    return proc, port

# ─── Helper: start Playwright Chromium ────────────────────────────────────────

async def start_playwright(display: int, url: str):
    os.environ["GOOGLE_API_KEY"] = "no"
    os.environ["GOOGLE_DEFAULT_CLIENT_ID"] = "no"
    os.environ["GOOGLE_DEFAULT_CLIENT_SECRET"] = "no"
    env = os.environ.copy()
    env["DISPLAY"] = f":{display}"

    # 1) Make a temporary directory for Chromium’s user data (so it’s a persistent context)
    user_data_dir = tempfile.mkdtemp(prefix="pw-user-data-")

    # 2) Launch a persistent context. Chromium will open the --app=<URL> automatically in this context.
    playwright = await async_playwright().start()
    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=user_data_dir,
        headless=False,
        env=env,
        args=[
            "--disable-gpu",

            # Open directly in “app” mode (no tabs, no address bar)
            "--app=" + url,

            # Force it to fill the entire VNC desktop (1280×720):
            "--window-size=1280,720",
            "--window-position=0,0",

            "--disable-infobars",
            "--class=BorderlessChromium",
            # "--disable-blink-features=AutomationControlled",
            "--disable-features=AutomationControlled",
            '--start-fullscreen',
            '--start-maximized',
            '--disable-translate',
            '--disable-dev-shm-usage'
        ],
        ignore_default_args=["--enable-automation", "--no-sandbox"],
    )


    # 3) Now that it’s persistent, the “app” page already exists in context.pages
    #    If it’s not there for some reason, do a fallback new_page()
    if context.pages:
        page = context.pages[0]
    else:
        page = await context.new_page()
        await page.goto(url)

    # 4) Wait for the DOM to be ready before sending F11
    await page.wait_for_load_state("domcontentloaded")
    await page.keyboard.press("F11")

    return playwright, context


# ─── Main: orchestrate everything ─────────────────────────────────────────────

async def main():
    # 1) Check/install dependencies
    check_and_install_dependencies()

    display = None
    xvnc_proc = None
    flux_proc = None
    novnc_proc = None
    playwright = None
    browser = None

    try:
        # 2) Find a free display, start Xvnc on it
        display, xvnc_proc = await find_free_display()
        print(f"Using display :{display} (Xvnc PID {xvnc_proc.pid})")

        # 3) Start fluxbox
        flux_proc = await start_window_manager(display)

        # 4) Start noVNC to proxy VNC → HTTP
        novnc_proc, novnc_port = await start_novnc(display)
        print(
            f"noVNC URL → "
            f"http://localhost:{novnc_port}/vnc.html?host=localhost&port={novnc_port}&autoconnect=true"
        )

        # 5) Launch Playwright→Chromium inside DISPLAY=:N
        playwright, browser = await start_playwright(display, "https://w3schools.com")

        # 6) Block until Ctrl+C
        await asyncio.Event().wait()

    except KeyboardInterrupt:
        print("\nCaught Ctrl+C → cleaning up…")

    finally:
        # 7) Cleanup: close Playwright
        if browser and playwright:
            try:
                await browser.close()
                await playwright.stop()
            except Exception:
                pass

        # 8) Terminate subprocesses (reverse order)
        for name, proc in [
            ("noVNC proxy", novnc_proc),
            ("fluxbox", flux_proc),
            ("Xvnc", xvnc_proc)
        ]:
            if proc:
                print(f"Terminating {name} (PID {proc.pid})…")
                try:
                    proc.terminate()
                    await asyncio.sleep(0.1)
                except Exception:
                    pass

        # 9) Kill vncserver on that display to clean up ~/.vnc
        if display is not None:
            print(f"Killing vncserver on :{display}…")
            subprocess.run(
                ["vncserver", "-kill", f":{display}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    asyncio.run(main())
