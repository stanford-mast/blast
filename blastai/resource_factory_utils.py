"""Utilities for resource factory operations."""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from browser_use.browser import BrowserSession, BrowserProfile
from patchright.async_api import async_playwright as async_patchright
from playwright.async_api import async_playwright

from .cli_installation import _get_novnc_path
from .vnc_utils import (
    _vnc_lock, _active_displays,
    _check_port_in_use, _find_available_display,
    _configure_xstartup, _configure_openbox,
    _kill_port_process
)

logger = logging.getLogger(__name__)

class VNCSession:
    """Manages a VNC session with browser control for human-in-loop automation.
    
    Uses X11 displays in range 1-20. For each display N:
    - VNC server runs on port 5900 + N
    - noVNC websocket proxy runs on port 6080 + N
    """
    
    def __init__(self, display_no: Optional[int] = None, geometry: str = "1280x720",
                 user_data_dir: Optional[str] = None, require_patchright: bool = True,
                 initial_url: str = "https://google.com"):
        """Initialize VNC session.
        
        Args:
            display_no: Optional display number (auto-assigned if None)
            geometry: Screen resolution (width x height)
            user_data_dir: Optional browser profile directory
            require_patchright: Whether to use patchright for stealth
            initial_url: Initial URL to load
            
        Raises:
            RuntimeError: If display number is unavailable or setup fails
        """
        try:
            # Check platform support first
            if sys.platform == 'win32':
                logger.error("VNC support not available on Windows")
                raise RuntimeError(
                    "VNC support is only available on Linux and MacOS systems. "
                    "For Windows users, please use WSL or a Linux virtual machine."
                )
            
            # Find first available display number if not specified
            if display_no is None:
                display_no = _find_available_display()
                if display_no is None:
                    raise RuntimeError("No available display numbers")
            elif display_no in _active_displays or _check_port_in_use(5900 + display_no) or _check_port_in_use(6080 + display_no):
                raise RuntimeError(f"Display :{display_no} or its ports are already in use")
            else:
                _active_displays.add(display_no)
            
            self.display_no = display_no
            self.geometry = geometry
            self.vnc_port = 5900 + display_no
            self.http_port = 6080 + display_no
            self.web_proc = None
            self.browser_session = None
            self.user_data_dir = user_data_dir
            self.require_patchright = require_patchright
            self.initial_url = initial_url
            
            # Set up VNC environment
            try:
                # Configure VNC and window manager
                _configure_xstartup(self.display_no)
                _configure_openbox(self.display_no)
            except Exception as e:
                _active_displays.remove(display_no)
                raise RuntimeError(f"Failed to set up VNC environment: {e}") from e
                
        except Exception as e:
            logger.error(f"Failed to initialize VNC session: {e}")
            raise
        
        
    async def start(self) -> Optional[Tuple[str, BrowserSession]]:
        """Start VNC session and browser session.
        
        Returns:
            Optional[Tuple[str, BrowserSession]]: Tuple of (live_url, browser_session) or None if failed
            
        Raises:
            RuntimeError: If no display is available or VNC server fails to start
        """
        try:
            # Start VNC server with synchronized operations
            with _vnc_lock:
                # Start Xvfb
                logger.debug(f"Starting Xvfb on display :{self.display_no}")
                # On MacOS, XQuartz provides the X server
                if sys.platform == 'darwin':
                    xvfb_cmd = f"Xvfb :{self.display_no} -screen 0 {self.geometry}x24 -retro"
                else:
                    xvfb_cmd = f"Xvfb :{self.display_no} -screen 0 {self.geometry}x24"
                self.xvfb_proc = subprocess.Popen(
                    xvfb_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ.copy(),
                    text=True
                )
                
                # Wait for Xvfb to start
                max_attempts = 10
                for attempt in range(max_attempts):
                    if self.xvfb_proc.poll() is not None:
                        stdout, stderr = self.xvfb_proc.communicate()
                        logger.error(f"Xvfb failed to start. stdout: {stdout}, stderr: {stderr}")
                        raise RuntimeError("Xvfb process terminated unexpectedly")
                        
                    try:
                        result = subprocess.run(
                            f"{'xdpyinfo' if sys.platform == 'linux' else '/opt/X11/bin/xdpyinfo'} -display :{self.display_no}",
                            shell=True,
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        # logger.debug(f"Xvfb check output: {result.stdout}")
                        break
                    except subprocess.CalledProcessError as e:
                        logger.debug(f"Waiting for Xvfb, attempt {attempt + 1}: {e.stderr}")
                        if attempt == max_attempts - 1:
                            raise RuntimeError("Failed to start Xvfb")
                        time.sleep(0.5)
                
                # Set display environment variable
                os.environ["DISPLAY"] = f":{self.display_no}"
                
                # Start VNC server
                logger.debug(f"Starting VNC server on display :{self.display_no}")
                # On MacOS, x11vnc needs additional options for XQuartz
                if sys.platform == 'darwin':
                    vnc_cmd = f"x11vnc -display :{self.display_no} -geometry {self.geometry} -forever -shared -rfbport {self.vnc_port} -nopw -listen localhost -xkb -noxrecord -noxfixes -noxdamage"
                else:
                    vnc_cmd = f"x11vnc -display :{self.display_no} -geometry {self.geometry} -forever -shared -rfbport {self.vnc_port} -nopw -listen localhost"
                self.vnc_proc = subprocess.Popen(
                    vnc_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ.copy(),
                    text=True
                )
                
                # Check VNC server started
                if self.vnc_proc.poll() is not None:
                    stdout, stderr = self.vnc_proc.communicate()
                    logger.error(f"VNC server failed to start. stdout: {stdout}, stderr: {stderr}")
                    raise RuntimeError("VNC server process terminated unexpectedly")
                    
                logger.debug("VNC server started successfully")
                
                # Wait for VNC server
                max_attempts = 10
                for attempt in range(max_attempts):
                    if _check_port_in_use(self.vnc_port):
                        break
                    if attempt == max_attempts - 1:
                        raise RuntimeError("Failed to start VNC server")
                    time.sleep(0.5)
                
                # Start noVNC proxy
                logger.debug(f"Starting noVNC proxy on port {self.http_port}")
                # Start noVNC proxy with output logging
                logger.debug(f"Starting noVNC proxy on port {self.http_port}")
                self.web_proc = subprocess.Popen(
                    f"websockify --web {_get_novnc_path()} {self.http_port} localhost:{self.vnc_port}",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ.copy(),
                    text=True
                )
                
                # Check if process started successfully
                if self.web_proc.poll() is not None:
                    stdout, stderr = self.web_proc.communicate()
                    logger.error(f"noVNC proxy failed to start. stdout: {stdout}, stderr: {stderr}")
                    raise RuntimeError("noVNC proxy process terminated unexpectedly")
                
                logger.debug("noVNC proxy process started")
                
                # Wait for proxy port to be available
                max_attempts = 10
                for attempt in range(max_attempts):
                    if _check_port_in_use(self.http_port):
                        logger.debug("noVNC proxy port is ready")
                        break
                        
                    # Check if process died while waiting
                    if self.web_proc.poll() is not None:
                        stdout, stderr = self.web_proc.communicate()
                        logger.error(f"noVNC proxy terminated while waiting. stdout: {stdout}, stderr: {stderr}")
                        raise RuntimeError("noVNC proxy process terminated while waiting for port")
                        
                    if attempt == max_attempts - 1:
                        logger.error("Timed out waiting for noVNC proxy port")
                        raise RuntimeError("Failed to start noVNC proxy (port timeout)")
                        
                    logger.debug(f"Waiting for noVNC proxy port, attempt {attempt + 1}")
                    time.sleep(0.5)
                
                # Get live URL
                live_url = f"http://localhost:{self.http_port}/vnc.html"
            
            # Parse geometry into width/height
            width, height = map(int, self.geometry.split('x'))
            
            # Create browser profile with all settings
            browser_profile = BrowserProfile(
                headless=False,  # Must be windowed mode for VNC
                user_data_dir=self.user_data_dir or f'/tmp/playwright_{self.display_no}',
                window_size={"width": width, "height": height},  # Match VNC geometry
                viewport=None,  # Let window size control viewport
                no_viewport=True,  # Disable fixed viewport in windowed mode
                keep_alive=True,  # Keep browser alive between tasks
                args=[
                    f'--app={self.initial_url}',
                    '--disable-infobars',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=TranslateUI,OverlayScrollbar,ExperimentalFullscreenExitUI',
                    '--kiosk',
                    '--start-fullscreen',
                    '--start-maximized',
                    '--disable-translate',
                    '--disable-dev-shm-usage'
                ],
                ignore_default_args=['--enable-automation'],
                env={"DISPLAY": os.environ["DISPLAY"]},
                disable_security=False,
                deterministic_rendering=False
            )
            
            # Create browser session
            self.browser_session = BrowserSession(
                browser_profile=browser_profile,
                playwright=await (async_patchright().start() if self.require_patchright else async_playwright().start())
            )
            
            # Start browser session
            await self.browser_session.start()
            
            # Create and navigate initial page
            page = await self.browser_session.get_current_page()
            await page.goto(self.initial_url)
            
            # Ensure fullscreen
            result = subprocess.run(
                f"xdotool search --{'class' if sys.platform == 'linux' else 'name'} " +
                ('chromium' if sys.platform == 'linux' else 'chromium') +
                " windowactivate --sync key F11",
                shell=True,
                capture_output=True,
                text=True,
                env=os.environ.copy()
            )
            if result.stdout:
                logger.debug(f"Xdotool output: {result.stdout}")
            if result.stderr:
                logger.debug(f"Xdotool error: {result.stderr}")
            
            return live_url, self.browser_session
            
        except Exception as e:
            logger.error(f"Failed to start VNC session: {e}")
            await self.cleanup()
            return None
        
    async def cleanup(self):
        """Clean up VNC session and browser session.
        
        Ensures all resources are properly cleaned up, including:
        - Browser session
        - VNC server
        - Port bindings
        - Web process
        - Display number
        - Config files
        """
        errors = []
        
        # Close browser session first
        if self.browser_session:
            try:
                await self.browser_session.stop()
            except Exception as e:
                errors.append(f"Failed to stop browser session: {e}")
            self.browser_session = None
        
        # Clean up VNC resources
        with _vnc_lock:
            try:
                # Kill Xvfb and VNC server
                try:
                    result = subprocess.run(
                        "pkill -f " + ("'Xvfb.*:" if sys.platform == 'linux' else "'X.*:") + f"{self.display_no}'",
                        shell=True,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logger.debug(f"Xvfb cleanup output: {result.stdout}")
                except subprocess.CalledProcessError as e:
                    errors.append(f"Failed to kill Xvfb: {e}")
                    
                try:
                    result = subprocess.run(
                        f"pkill -f 'x11vnc.*:{self.vnc_port}'",
                        shell=True,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logger.debug(f"VNC server cleanup output: {result.stdout}")
                except subprocess.CalledProcessError as e:
                    errors.append(f"Failed to kill VNC server: {e}")
            except subprocess.CalledProcessError as e:
                errors.append(f"Failed to kill VNC server: {e}")
                
            # Kill ports
            for port in (self.vnc_port, self.http_port):
                try:
                    if _check_port_in_use(port):
                        _kill_port_process(port)
                except subprocess.CalledProcessError as e:
                    errors.append(f"Failed to kill port {port}: {e}")
                    
            # Kill web process
            if self.web_proc:
                try:
                    self.web_proc.terminate()
                    self.web_proc.wait(timeout=5)
                except Exception as e:
                    errors.append(f"Failed to terminate web process: {e}")
                    try:
                        self.web_proc.kill()
                    except Exception:
                        pass
                self.web_proc = None
                
            # Clean up config files
            try:
                # Clean up display-specific xstartup
                xstartup_path = Path.home() / ".vnc" / f"xstartup.{self.display_no}"
                if xstartup_path.exists():
                    xstartup_path.unlink()
                    
                # Clean up default symlink if it points to our file
                default_path = Path.home() / ".vnc" / "xstartup"
                if default_path.is_symlink():
                    target = default_path.resolve()
                    if str(target).endswith(f"xstartup.{self.display_no}"):
                        default_path.unlink()
            except Exception as e:
                errors.append(f"Failed to clean up xstartup: {e}")
                
            # Always remove from active displays
            if self.display_no in _active_displays:
                _active_displays.remove(self.display_no)
        
        if errors:
            logger.error("Errors during VNC cleanup:\n" + "\n".join(errors))

def get_stealth_profile_dir(task_id: str) -> str:
    """Get a unique stealth profile directory path for a task.
    
    Args:
        task_id: Task ID to create profile for
        
    Returns:
        Path to stealth profile directory (will be created if doesn't exist)
    """
    # Get base and task-specific paths
    base_stealth_dir = os.path.expanduser('~/.config/browseruse/profiles/stealth')
    stealth_dir = os.path.expanduser(f'~/.config/browseruse/profiles/stealth_{task_id}')
    
    base_stealth_path = Path(base_stealth_dir)
    stealth_dir_path = Path(stealth_dir)
    
    # Ensure base directory exists
    if not base_stealth_path.exists():
        return base_stealth_dir
    
    # Create task-specific directory
    if stealth_dir_path.exists():
        shutil.rmtree(stealth_dir)
    if base_stealth_path.exists():
        shutil.copytree(base_stealth_dir, stealth_dir)
    else:
        stealth_dir_path.mkdir(parents=True, exist_ok=True)
    
    return stealth_dir

def cleanup_stealth_profile_dir(profile_dir: str) -> None:
    """Safely clean up a stealth profile directory.
    
    Args:
        profile_dir: Path to profile directory to clean up
    """
    try:
        if not profile_dir or 'stealth_' not in profile_dir:
            return
            
        profile_path = Path(os.path.expanduser(profile_dir))
        if profile_path.exists():
            shutil.rmtree(profile_path)
            logger.debug(f"Cleaned up stealth profile: {profile_path}")
    except Exception as e:
        logger.error(f"Error cleaning up stealth profile: {e}")