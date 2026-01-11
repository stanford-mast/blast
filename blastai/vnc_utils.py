"""VNC session management utilities."""

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Global lock for VNC operations
_vnc_lock = threading.Lock()

# Track active display numbers
_active_displays: Set[int] = set()


def _get_port_command(port: int, action: str = "check") -> str:
    """Get platform-specific command for port operations.

    Args:
        port: Port number
        action: Either 'check' or 'kill'

    Returns:
        Command string for the specified action

    Raises:
        ValueError: If action is invalid
        RuntimeError: If platform is not supported
    """
    if sys.platform == "darwin":
        if action == "check":
            return f"lsof -i :{port}"
        elif action == "kill":
            return f"lsof -ti :{port} | xargs kill -9"
    elif sys.platform == "linux":
        if action == "check":
            return f"fuser {port}/tcp"
        elif action == "kill":
            return f"fuser -k {port}/tcp"
    else:
        raise RuntimeError("Platform not supported")

    if action not in ("check", "kill"):
        raise ValueError("Action must be 'check' or 'kill'")


def _kill_port_process(port: int) -> None:
    """Kill process using a specific port.

    Uses platform-specific commands:
    - Linux: Uses fuser
    - MacOS: Uses lsof + kill

    Args:
        port: Port number to kill process for

    Raises:
        subprocess.CalledProcessError: If command fails
    """
    cmd = _get_port_command(port, "kill")
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    logger.debug(f"Port {port} cleanup output: {result.stdout}")


def _check_port_in_use(port: int) -> bool:
    """Check if a TCP port is in use.

    Uses platform-specific commands to check port status:
    - Linux: Uses fuser
    - MacOS: Uses lsof
    - Windows: Not supported

    Args:
        port: Port number to check

    Returns:
        bool: True if port is in use, False otherwise
    """
    # Try socket bind first (works on all platforms)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return False
    except socket.error:
        return True
    except Exception as e:
        logger.debug(f"Socket bind check failed: {e}")
        # Fall back to platform-specific checks

    # Platform-specific checks as backup
    try:
        cmd = _get_port_command(port, "check")
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        logger.debug(f"Port {port} check output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.debug(f"Port {port} check error: {e.stderr}")
        return False


def _find_available_display() -> Optional[int]:
    """Find first available display number and associated ports.

    Searches for an available display number in range 1-20 (standard X11 display range).
    For each display number N:
    - VNC server uses port 5900 + N
    - noVNC websocket proxy uses port 6080 + N

    Returns:
        Optional[int]: Available display number or None if none available
    """
    with _vnc_lock:
        # Try displays 1-20 (standard X11 display range)
        for display_no in range(1, 21):
            if display_no in _active_displays:
                continue

            vnc_port = 5900 + display_no
            http_port = 6080 + display_no

            # Check if ports are in use
            if not _check_port_in_use(vnc_port) and not _check_port_in_use(http_port):
                _active_displays.add(display_no)
                return display_no

        return None


def _configure_xstartup(display_no: int) -> None:
    """Configure .vnc/xstartup for the display."""
    with _vnc_lock:
        xstartup_dir = Path.home() / ".vnc"
        xstartup_dir.mkdir(parents=True, exist_ok=True)

        # Clean up any existing files for this display
        xstartup_path = xstartup_dir / f"xstartup.{display_no}"
        if xstartup_path.exists():
            xstartup_path.unlink()

        # Create new xstartup file
        xstartup_content = """#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec openbox-session
"""
        xstartup_path.write_text(xstartup_content)
        xstartup_path.chmod(0o755)

        # Handle default xstartup symlink
        default_path = xstartup_dir / "xstartup"
        try:
            # Remove existing symlink or file
            if default_path.is_symlink() or default_path.exists():
                default_path.unlink()
            # Create new symlink
            default_path.symlink_to(xstartup_path)
        except Exception as e:
            logger.error(f"Failed to create xstartup symlink: {e}")
            # Continue anyway since the display-specific file exists


def _configure_openbox(display_no: int) -> None:
    """Configure openbox for the display."""
    with _vnc_lock:
        config_dir = Path.home() / f".config/openbox"
        config_dir.mkdir(parents=True, exist_ok=True)

        rc_path = config_dir / "rc.xml"
        rc_content = """<?xml version="1.0" encoding="UTF-8"?>
<openbox_config>
  <applications>
    <application class="*">
      <decor>no</decor>
      <maximized>true</maximized>
    </application>
  </applications>
</openbox_config>
"""
        rc_path.write_text(rc_content)
