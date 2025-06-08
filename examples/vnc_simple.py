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

# Global set to track allocated display numbers
allocated_displays = set()

# ─── Helper: find a free display & start Xvnc ─────────────────────────────────

async def find_free_display():
    """
    Iterate through display numbers 1..99. For each:
      1. Skip if display is already allocated to a running session
      2. Kill any existing Xvnc (vncserver) on :N.
      3. Remove stale ~/.vnc/X<N>.* files.
      4. Attempt to start Xvnc :N. If it stays alive for ~1s, return (N, proc).
    """
    home = Path(os.environ["HOME"])
    vnc_dir = home / ".vnc"
    vnc_dir.mkdir(exist_ok=True)

    for n in range(1, 100):
        # Skip if display is already allocated
        if n in allocated_displays:
            continue

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
            allocated_displays.add(n)  # Mark display as allocated
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
    Return first free TCP port in [start..end].
    """
    for port in range(start, end + 1):
        with socket.socket() as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free HTTP port between {start} and {end}")


async def setup_novnc_session(display: int) -> Path:
    """Create a session-specific noVNC directory with a patched vnc.html"""
    base_dir = Path.home() / "noVNC"
    session_dir = Path.home() / f"noVNC_session_{display}"
    
    # Copy the entire noVNC directory for this session
    if not session_dir.exists():
        import shutil
        shutil.copytree(base_dir, session_dir)
        
    # Patch the UI in this session's copy
    html_path = session_dir / "vnc.html"
    patch_version = "v0.1.22"

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
        return session_dir

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
    return session_dir

async def start_novnc(display: int):
    """
    Launch noVNC's novnc_proxy, forwarding VNC (5900+display) → HTTP.
    Returns (proc, http_port).
    """
    vnc_port = 5900 + display
    http_base = 6080  # Base HTTP port
    initial_port = http_base + (display - 1)  # Offset by display number
    
    # Setup a session-specific noVNC directory
    novnc_dir = await setup_novnc_session(display)
    proxy = novnc_dir / "utils" / "novnc_proxy"

    port = await find_free_http_port(initial_port, initial_port + 4)  # Smaller range per display
    cmd = ["bash", str(proxy), "--vnc", f"localhost:{vnc_port}", "--web", str(novnc_dir), "--listen", str(port)]
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


class VNCSession:
    def __init__(self, display, xvnc_proc, flux_proc, novnc_proc, novnc_port, playwright, browser):
        self.display = display
        self.xvnc_proc = xvnc_proc
        self.flux_proc = flux_proc
        self.novnc_proc = novnc_proc
        self.novnc_port = novnc_port
        self.playwright = playwright
        self.browser = browser
        self.novnc_dir = Path.home() / f"noVNC_session_{display}"

    async def cleanup(self):
        """Cleanup all processes associated with this VNC session"""
        # Close Playwright
        if self.browser and self.playwright:
            try:
                await self.browser.close()
                await self.playwright.stop()
            except Exception:
                pass

        # Terminate subprocesses
        for name, proc in [
            ("noVNC proxy", self.novnc_proc),
            ("fluxbox", self.flux_proc),
            ("Xvnc", self.xvnc_proc)
        ]:
            if proc:
                print(f"Terminating {name} (PID {proc.pid})…")
                try:
                    proc.terminate()
                    await asyncio.sleep(0.1)
                except Exception:
                    pass

        # Kill vncserver
        print(f"Killing vncserver on :{self.display}…")
        subprocess.run(
            ["vncserver", "-kill", f":{self.display}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Remove display from allocated set
        allocated_displays.remove(self.display)

        # Clean up session-specific noVNC directory
        import shutil
        if self.novnc_dir.exists():
            shutil.rmtree(self.novnc_dir)

async def launch_session(target_url: str) -> VNCSession:
    """Launch a VNC session with browser on a free display"""
    # 1) Find a free display, start Xvnc on it
    display, xvnc_proc = await find_free_display()
    print(f"Using display :{display} (Xvnc PID {xvnc_proc.pid})")

    # 2) Start fluxbox
    flux_proc = await start_window_manager(display)

    # 3) Start noVNC to proxy VNC → HTTP
    novnc_proc, novnc_port = await start_novnc(display)
    print(
        f"noVNC URL → "
        f"http://localhost:{novnc_port}/vnc.html?host=localhost&port={novnc_port}&autoconnect=true"
    )

    # 4) Launch Playwright→Chromium inside DISPLAY=:N
    playwright, browser = await start_playwright(display, target_url)

    return VNCSession(
        display=display,
        xvnc_proc=xvnc_proc,
        flux_proc=flux_proc,
        novnc_proc=novnc_proc,
        novnc_port=novnc_port,
        playwright=playwright,
        browser=browser
    )

async def main():
    # Check/install dependencies
    check_and_install_dependencies()

    sessions = []
    try:
        # Launch multiple sessions with different URLs
        sessions.append(await launch_session("https://example.com"))
        sessions.append(await launch_session("https://w3schools.com"))
        
        print("\nAll sessions are now running!")
        print("Press Ctrl+C to terminate all sessions")
        
        # Block until Ctrl+C
        await asyncio.Event().wait()

    except KeyboardInterrupt:
        print("\nCaught Ctrl+C → cleaning up…")

    finally:
        # Cleanup all sessions
        for session in sessions:
            await session.cleanup()
        print("All sessions cleaned up. Exiting.")

if __name__ == "__main__":
    asyncio.run(main())
