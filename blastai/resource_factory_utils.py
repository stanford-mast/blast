"""Utilities for BLAST resource factory including VNC support."""

import asyncio
import os
import re
import shutil
import socket
import subprocess
import sys
import platform
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

from browser_use import BrowserSession, BrowserProfile
from patchright.async_api import async_playwright as async_patchright

import logging
logger = logging.getLogger(__name__)

# Global set to track allocated display numbers
allocated_displays = set()

def get_stealth_profile_dir(task_id: str) -> str:
    """Get a unique stealth profile directory path for a task."""
    base_stealth_dir = os.path.expanduser('~/.config/browseruse/profiles/stealth')
    stealth_dir = os.path.expanduser(f'~/.config/browseruse/profiles/stealth_{task_id}')
    
    base_stealth_path = Path(base_stealth_dir)
    stealth_dir_path = Path(stealth_dir)
    
    # Ensure base directory exists
    if not base_stealth_path.exists():
        return base_stealth_dir
    
    # Create task-specific directory
    if stealth_dir_path.exists():
        try:
            shutil.rmtree(stealth_dir)
        except OSError as e:
            logger.warning(f"Failed to remove existing stealth directory: {e}")
            # Try to create a unique directory instead
            stealth_dir = os.path.expanduser(f'~/.config/browseruse/profiles/stealth_{task_id}_{int(time.time())}')
            stealth_dir_path = Path(stealth_dir)
    
    if base_stealth_path.exists():
        try:
            shutil.copytree(base_stealth_dir, stealth_dir)
        except OSError as e:
            logger.warning(f"Failed to copy stealth directory: {e}")
            # Just create an empty directory
            stealth_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        stealth_dir_path.mkdir(parents=True, exist_ok=True)
    
    return stealth_dir

def cleanup_stealth_profile_dir(profile_dir: str) -> None:
    """Safely clean up a stealth profile directory."""
    try:
        if not profile_dir or 'stealth_' not in profile_dir:
            return
            
        profile_path = Path(os.path.expanduser(profile_dir))
        if profile_path.exists():
            shutil.rmtree(profile_path)
            logger.debug(f"Cleaned up stealth profile: {profile_path}")
    except Exception as e:
        logger.error(f"Error cleaning up stealth profile: {e}")

async def find_free_display() -> Tuple[int, asyncio.subprocess.Process]:
    """Find a free display number and start Xvnc on it."""
    home = Path(os.environ["HOME"])
    vnc_dir = home / ".vnc"
    vnc_dir.mkdir(exist_ok=True)

    for n in range(1, 100):
        # Skip if display is already allocated
        if n in allocated_displays:
            continue

        # Kill any existing Xvnc
        # Redirect vncserver output to logger
        result = subprocess.run(
            ["vncserver", "-kill", f":{n}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        if result.stdout:
            logger.debug(result.stdout.strip())
        if result.stderr:
            logger.debug(result.stderr.strip())
        # Remove leftover files
        for suffix in [f"X{n}.sock", f"{home.name}:{n}.pid", f"{home.name}:{n}.log"]:
            try:
                (vnc_dir / suffix).unlink(missing_ok=True)
            except PermissionError:
                pass

        # Try starting Xvnc
        try:
            xvnc_proc = await start_xvnc(n)
            allocated_displays.add(n)  # Mark display as allocated
            return n, xvnc_proc
        except Exception as e:
            logger.debug(f"Display :{n} failed: {e}")
            continue

    raise RuntimeError("No free display found")

async def start_xvnc(display: int) -> asyncio.subprocess.Process:
    """Launch Xvnc on the specified display."""
    cmd = [
        "Xvnc",
        f":{display}",
        "-geometry", "1280x720",
        "-depth", "24",
        "-SecurityTypes", "None"
    ]
    logger.debug(f"[Xvnc] {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Poll for success
    for _ in range(10):
        await asyncio.sleep(0.1)
        if proc.returncode not in (None, 0):
            raise RuntimeError(f"Xvnc exited with code {proc.returncode}")
    return proc

async def start_window_manager(display: int) -> asyncio.subprocess.Process:
    """Start window manager on the given display."""
    system = platform.system()
    env = os.environ.copy()
    env["DISPLAY"] = f":{display}"
    
    if system == "Linux":
        # Use matchbox-window-manager to remove all decorations
        wm_cmd = ["matchbox-window-manager", "-use_titlebar", "no"]
        logger.debug(f"[WM] Running (Linux): {' '.join(wm_cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *wm_cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await asyncio.sleep(0.5)
        return proc
    
    elif system == "Darwin":
        # On macOS, matchbox isn't available. Fall back to fluxbox
        wm_cmd = ["fluxbox", "-display", f":{display}"]
        logger.debug(f"[WM] Running (macOS fallback): {' '.join(wm_cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *wm_cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await asyncio.sleep(0.5)
        return proc
    
    else:
        # If some other OS, just attempt fluxbox
        wm_cmd = ["fluxbox", "-display", f":{display}"]
        logger.debug(f"[WM] Running (fallback): {' '.join(wm_cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *wm_cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await asyncio.sleep(0.5)
        return proc

async def find_free_http_port(start: int = 6080, end: int = 6099) -> int:
    """Find a free HTTP port in the given range."""
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
        logger.debug(f"noVNC UI already patched with {patch_version}.")
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

    logger.debug(f"Patched {html_path} with {patch_version}")
    return session_dir

async def start_novnc(display: int) -> Tuple[asyncio.subprocess.Process, int, Path]:
    """Start noVNC proxy for the given display."""
    vnc_port = 5900 + display
    http_base = 6080  # Base HTTP port
    initial_port = http_base + (display - 1)  # Offset by display number
    
    # Setup a session-specific noVNC directory
    novnc_dir = await setup_novnc_session(display)
    proxy = novnc_dir / "utils" / "novnc_proxy"

    port = await find_free_http_port(initial_port, initial_port + 4)  # Smaller range per display
    
    # Add parameters to prevent WebSocket connection from timing out
    cmd = [
        "bash", str(proxy),
        "--vnc", f"localhost:{vnc_port}",
        "--web", str(novnc_dir),
        "--listen", str(port),
        "--heartbeat", "15",  # Send a ping every 15 seconds to keep connection alive
        "--idle-timeout", "86400",  # 24 hours idle timeout
        "--timeout", "0"  # No timeout (run indefinitely)
    ]
    
    logger.debug(f"[noVNC] {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Wait a bit longer to ensure the proxy is fully started
    await asyncio.sleep(1.0)
    
    return proc, port, novnc_dir

class VNCSession:
    """Manages a VNC session with browser integration."""
    
    def __init__(self, display: int, xvnc_proc: asyncio.subprocess.Process,
                 wm_proc: asyncio.subprocess.Process, novnc_proc: asyncio.subprocess.Process,
                 novnc_port: int, browser_session: BrowserSession, page: any,
                 stealth: bool = False, novnc_dir: Optional[Path] = None):
        """Initialize VNC session with all components."""
        self.display = display
        self.xvnc_proc = xvnc_proc
        self.wm_proc = wm_proc
        self.novnc_proc = novnc_proc
        self.novnc_port = novnc_port
        self.browser_session = browser_session
        self.page = page
        self.stealth = stealth
        self.novnc_dir = novnc_dir
        self.stealth_dir = get_stealth_profile_dir(f"vnc_{display}") if stealth else None

    async def get_browser_session(self) -> BrowserSession:
        """Get the browser session associated with this VNC session."""
        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")
        return self.browser_session

    async def cleanup(self) -> None:
        """Clean up all resources associated with the VNC session."""
        # Close browser session
        if self.browser_session:
            try:
                await self.browser_session.close()
            except Exception as e:
                logger.error(f"Error closing browser session: {e}")

        # Terminate subprocesses
        for name, proc in [
            ("noVNC proxy", self.novnc_proc),
            ("window manager", self.wm_proc),
            ("Xvnc", self.xvnc_proc)
        ]:
            if proc:
                logger.debug(f"Terminating {name} (PID {proc.pid})...")
                try:
                    proc.terminate()
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error terminating {name}: {e}")

        # Kill vncserver
        logger.debug(f"Killing vncserver on :{self.display}...")
        result = subprocess.run(
            ["vncserver", "-kill", f":{self.display}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        if result.stdout:
            logger.debug(result.stdout.strip())
        if result.stderr:
            logger.debug(result.stderr.strip())

        # Remove display from allocated set
        allocated_displays.remove(self.display)

        # Clean up session-specific noVNC directory
        if self.novnc_dir and self.novnc_dir.exists():
            shutil.rmtree(self.novnc_dir)

        # Clean up stealth profile if used
        if self.stealth_dir:
            cleanup_stealth_profile_dir(self.stealth_dir)

    def get_novnc_url(self) -> str:
        """Get the URL for accessing the noVNC web interface."""
        return f"http://localhost:{self.novnc_port}/vnc.html?autoconnect=true"

    async def __aenter__(self) -> 'VNCSession':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()

async def launch_vnc_session(target_url: str, stealth: bool = False) -> VNCSession:
    """Launch a complete VNC session with browser."""
    # 1) Find a free display, start Xvnc on it
    display, xvnc_proc = await find_free_display()
    logger.debug(f"Using display :{display} (Xvnc PID {xvnc_proc.pid})")

    try:
        # 2) Start window manager
        wm_proc = await start_window_manager(display)

        # 3) Start noVNC
        novnc_proc, novnc_port, novnc_dir = await start_novnc(display)
        logger.debug(f"noVNC URL: http://localhost:{novnc_port}/vnc.html?autoconnect=true")

        # 4) Set up browser environment
        env = os.environ.copy()
        env["DISPLAY"] = f":{display}"
        env["GOOGLE_API_KEY"] = "no"
        env["GOOGLE_DEFAULT_CLIENT_ID"] = "no"
        env["GOOGLE_DEFAULT_CLIENT_SECRET"] = "no"

        # 5) Configure browser
        browser_args = {
            'headless': False,
            'highlight_elements': False,  # Disable element highlighting
            'keep_alive': False,  # Set to False so agent.close() can clean up
            'env': env,
            'args': [
                "--disable-gpu",
                f"--app={target_url}",
                "--window-size=1280,720",
                "--window-position=0,0",
                "--disable-infobars",
                "--class=BorderlessChromium",
                "--disable-features=AutomationControlled",
                '--start-fullscreen',
                '--start-maximized',
                '--disable-translate',
                '--disable-dev-shm-usage'
            ],
            'ignore_default_args': ["--enable-automation", "--no-sandbox"],
        }

        if stealth:
            # Use patchright and stealth profile
            playwright = await async_patchright().start()
            browser_args['playwright'] = playwright
            stealth_dir = get_stealth_profile_dir(f"vnc_{display}")
            browser_args['user_data_dir'] = stealth_dir
            browser_args['disable_security'] = False
            browser_args['deterministic_rendering'] = False
        else:
            # Use regular temporary profile
            browser_args['user_data_dir'] = tempfile.mkdtemp(prefix="pw-user-data-")

        # 6) Launch browser session
        browser_profile = BrowserProfile(**browser_args)
        browser_session = BrowserSession(browser_profile=browser_profile)
        
        # Ensure the Default directory is empty or doesn't exist before starting
        if stealth:
            default_dir = Path(stealth_dir) / "Default"
            if default_dir.exists():
                try:
                    # Try to remove the Default directory if it exists
                    shutil.rmtree(str(default_dir))
                except OSError as e:
                    logger.warning(f"Failed to clean up Default directory: {e}")
                    # Create a new unique stealth directory
                    new_stealth_dir = get_stealth_profile_dir(f"vnc_{display}_{int(time.time())}")
                    browser_args['user_data_dir'] = new_stealth_dir
                    browser_profile = BrowserProfile(**browser_args)
                    browser_session = BrowserSession(browser_profile=browser_profile)
        await browser_session.start()

        # 7) Set up page
        page = await browser_session.get_current_page()
        await page.goto(target_url)
        await page.wait_for_load_state("domcontentloaded")
        await page.keyboard.press("F11")

        # 8) Create and return session
        return VNCSession(
            display=display,
            xvnc_proc=xvnc_proc,
            wm_proc=wm_proc,
            novnc_proc=novnc_proc,
            novnc_port=novnc_port,
            browser_session=browser_session,
            page=page,
            stealth=stealth,
            novnc_dir=novnc_dir
        )

    except Exception as e:
        # Clean up on failure
        if display in allocated_displays:
            allocated_displays.remove(display)
        raise RuntimeError(f"Failed to launch VNC session: {e}") from e