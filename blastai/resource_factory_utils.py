"""Utilities for resource factory operations."""

import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Set
import re
import threading

logger = logging.getLogger(__name__)

# Global lock for VNC operations
_vnc_lock = threading.Lock()

# Track active display numbers
_active_displays: Set[int] = set()

class VNCBrowserSession:
    """Manages a VNC session with browser control for human-in-loop automation."""
    
    def __init__(self, display_no: int = None, geometry: str = "1280x720",
                 user_data_dir: Optional[str] = None, require_patchright: bool = True,
                 initial_url: str = "https://google.com"):
        """Initialize VNC browser session.
        
        Args:
            display_no: Optional display number (auto-assigned if None)
            geometry: Screen resolution
            user_data_dir: Optional browser profile directory
            require_patchright: Whether to use patchright for stealth
        """
        # Find first available display number if not specified
        if display_no is None:
            display_no = self._find_available_display()
            
        self.display_no = display_no
        self.geometry = geometry
        self.vnc_port = 5900 + display_no
        self.http_port = 6080 + display_no
        self.web_proc = None
        self.browser_context = None
        self.user_data_dir = user_data_dir
        self.require_patchright = require_patchright
        self.initial_url = initial_url
        self._setup_requirements()
        
    def _find_available_display(self) -> Optional[int]:
        """Find first available display number starting from 1.
        
        Returns:
            Optional[int]: Available display number or None if none available
        """
        with _vnc_lock:
            # Try up to display 20 (arbitrary limit to prevent infinite loop)
            for display_no in range(1, 21):
                if display_no in _active_displays:
                    continue
                    
                vnc_port = 5900 + display_no
                http_port = 6080 + display_no
                
                # Check if ports are in use
                vnc_in_use = subprocess.run(
                    f"fuser {vnc_port}/tcp",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                ).returncode == 0
                
                http_in_use = subprocess.run(
                    f"fuser {http_port}/tcp",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                ).returncode == 0
                
                if not vnc_in_use and not http_in_use:
                    _active_displays.add(display_no)
                    return display_no
                    
            return None
        
    def _setup_requirements(self):
        """Check and install required VNC packages."""
        required_packages = [
            "openbox",
            "tigervnc-standalone-server",
            "novnc",
            "websockify",
            "dbus-x11",
            "x11-utils",
            "xdotool"
        ]
        
        # Check which packages are missing
        missing_packages = []
        for package in required_packages:
            result = subprocess.run(
                f"dpkg -l {package} 2>/dev/null | grep -E '^ii'",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                missing_packages.append(package)
                
        if missing_packages:
            logger.warning(f"Missing VNC dependencies: {', '.join(missing_packages)}")
            logger.warning("Please install manually to enable VNC functionality")
            
    def _configure_xstartup(self):
        """Configure VNC xstartup file."""
        vnc_dir = Path.home() / ".vnc"
        vnc_dir.mkdir(exist_ok=True)
        xstartup = vnc_dir / "xstartup"
        xstartup.write_text("""#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
[ -r "$HOME/.Xresources" ] && xrdb "$HOME/.Xresources"
vncconfig -iconic &
exec dbus-launch --exit-with-session openbox-session
""")
        xstartup.chmod(0o755)
        
    def _configure_openbox(self):
        """Configure Openbox window manager."""
        ob_dir = Path.home() / ".config" / "openbox"
        ob_dir.mkdir(parents=True, exist_ok=True)
        rc = ob_dir / "rc.xml"
        rc.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<openbox_config>
  <applications>
    <application class="Chromium-browser">
      <decor>no</decor>
      <fullscreen>yes</fullscreen>
      <maximize>yes</maximize>
    </application>
  </applications>
  <theme>
    <titleLayout>NL</titleLayout>
    <keepBorder>no</keepBorder>
  </theme>
</openbox_config>
""")
        subprocess.run("openbox --reconfigure || true", shell=True)
        
    async def start(self) -> Optional[Tuple[str, Any]]:
        """Start VNC session and browser page.
        
        Returns:
            Optional[Tuple[str, Any]]: Tuple of (live_url, page) or None if failed
            
        Raises:
            RuntimeError: If no display is available or VNC server fails to start
        """
        try:
            # Find available display
            if self.display_no is None:
                self.display_no = self._find_available_display()
                if self.display_no is None:
                    raise RuntimeError("No available display numbers")
                    
            # Ensure VNC operations are synchronized
            with _vnc_lock:
                # Configure VNC and window manager
                self._configure_xstartup()
                self._configure_openbox()
                
                # Set display environment variable
                os.environ["DISPLAY"] = f":{self.display_no}"
                
                # Start VNC server
                subprocess.run(
                    f"vncserver :{self.display_no} -geometry {self.geometry} -localhost no -SecurityTypes None --I-KNOW-THIS-IS-INSECURE",
                    shell=True,
                    check=True,
                    env=os.environ.copy()
                )
                
                # Start noVNC proxy
                self.web_proc = subprocess.Popen(
                    f"websockify --web /usr/share/novnc {self.http_port} localhost:{self.vnc_port}",
                    shell=True,
                    env=os.environ.copy()
                )
                time.sleep(2)  # Wait for proxy to start
                
                # Get live URL
                live_url = f"http://localhost:{self.http_port}/vnc.html"
            
            # Launch browser context
            if self.require_patchright:
                from patchright.async_api import async_playwright
                playwright = await async_playwright().start()
            else:
                from playwright.async_api import async_playwright
                playwright = await async_playwright().start()
                
            # Configure browser launch args
            browser_args = [
                f'--app={self.initial_url}',
                '--disable-infobars',
                '--disable-blink-features=AutomationControlled',
                '--disable-features=TranslateUI,OverlayScrollbar,ExperimentalFullscreenExitUI',
                '--kiosk',
                '--start-fullscreen',
                '--start-maximized',
                '--disable-translate',
                '--disable-dev-shm-usage'
            ]
            
            # Launch persistent context
            user_data_dir = self.user_data_dir or f'/tmp/playwright_{self.display_no}'
            self.browser_context = await playwright.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=False,
                args=browser_args,
                ignore_default_args=['--enable-automation'],
                env=os.environ.copy()
            )
            
            # Create and navigate initial page
            page = await self.browser_context.new_page()
            await page.goto(self.initial_url)
            
            # Ensure fullscreen
            subprocess.run(
                "xdotool search --class chromium windowactivate --sync key F11",
                shell=True,
                env=os.environ.copy()
            )
            
            # Return live URL and page
            return live_url, page
            
        except Exception as e:
            logger.error(f"Failed to start VNC browser session: {e}")
            await self.cleanup()
            return None
    async def cleanup(self):
        """Clean up VNC session and browser context."""
        try:
            # Close browser context first to ensure clean browser shutdown
            if self.browser_context:
                await self.browser_context.close()
                self.browser_context = None
            
            # Ensure VNC cleanup is synchronized
            with _vnc_lock:
                if self.display_no in _active_displays:
                    _active_displays.remove(self.display_no)
                
                # Kill VNC server
                subprocess.run(
                    f"vncserver -kill :{self.display_no}",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Kill ports
                for port in (self.vnc_port, self.http_port):
                    subprocess.run(
                        f"fuser -k {port}/tcp",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    
                # Kill web process
                if self.web_proc:
                    self.web_proc.terminate()
                    self.web_proc = None
                
        except Exception as e:
            logger.error(f"Error cleaning up VNC browser session: {e}")

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