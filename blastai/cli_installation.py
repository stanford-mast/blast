"""Installation and dependency management for BLAST CLI."""

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .utils import get_appdata_dir

logger = logging.getLogger("web")


def find_executable(*names: str) -> Optional[str]:
    """Find the first available executable from the given names."""
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def check_node_installation() -> Optional[str]:
    """Check if Node.js is installed and available."""
    node_cmd = find_executable("node", "node.exe")
    if not node_cmd:
        logger.error("Node.js not found")
        # Show installation instructions (essential user guidance)
        logger.warning("""
Node.js not found. The web frontend requires Node.js.

To install Node.js:
1. Visit https://nodejs.org
2. Download and install the LTS version
3. Run 'node --version' to verify installation

Falling back to CLI frontend for now...
Once Node.js is installed, run 'blastai serve' again to use the web frontend
""")
        return None
    return node_cmd


def check_npm_installation() -> Optional[str]:
    """Check if npm is installed and available, and attempt to install if not found."""
    npm_cmd = find_executable("npm", "npm.cmd")
    if npm_cmd:
        return npm_cmd

    logger.info("npm not found, attempting to install Node.js and npm...")

    try:
        if sys.platform == "win32":
            # On Windows, show manual installation instructions
            logger.error("npm not found")
            logger.warning("""
Node.js/npm installation required. Please:
1. Visit https://nodejs.org
2. Download and run the Windows Installer (.msi)
3. Follow the installation wizard
4. Restart your terminal and try again
""")
            return None

        elif sys.platform == "darwin":
            # On macOS, try using Homebrew
            try:
                # Check if Homebrew is installed
                subprocess.run(["which", "brew"], check=True, capture_output=True)
                # Install Node.js (includes npm)
                subprocess.run(["brew", "install", "node"], check=True)
                npm_cmd = find_executable("npm", "npm.cmd")
                if npm_cmd:
                    logger.info("Successfully installed Node.js and npm")
                    return npm_cmd
            except subprocess.CalledProcessError:
                logger.error("Homebrew not found or installation failed")
                logger.warning("""
Failed to install Node.js/npm. Please install manually:
1. Visit https://nodejs.org
2. Download and run the macOS Installer (.pkg)
3. Restart your terminal and try again
""")
                return None

        else:  # Linux
            try:
                # Try apt-get first (Ubuntu/Debian)
                try:
                    subprocess.run(["sudo", "apt-get", "update"], check=True)
                    subprocess.run(["sudo", "apt-get", "install", "-y", "nodejs", "npm"], check=True)
                except subprocess.CalledProcessError:
                    # If apt-get fails, try dnf (Fedora)
                    subprocess.run(["sudo", "dnf", "install", "-y", "nodejs", "npm"], check=True)

                npm_cmd = find_executable("npm", "npm.cmd")
                if npm_cmd:
                    logger.info("Successfully installed Node.js and npm")
                    return npm_cmd

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install Node.js/npm: {e}")
                logger.warning("""
Failed to install Node.js/npm automatically. Please install manually:

Ubuntu/Debian:
    sudo apt-get update && sudo apt-get install -y nodejs npm

Fedora:
    sudo dnf install -y nodejs npm

Or using nvm (recommended):
1. curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
2. Restart terminal
3. nvm install --lts
""")
                return None

    except Exception as e:
        logger.error(f"Error during Node.js/npm installation: {e}")
        return None

    return npm_cmd


def check_installation_state() -> Dict[str, bool]:
    """Check if browsers and dependencies are already installed."""
    state_file = get_appdata_dir() / "installation_state.json"
    default_state = {"browsers_installed": False, "vnc_installed": False}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            return {**default_state, **state}
    return default_state


def save_installation_state(state_updates: Dict[str, bool]):
    """Save installation state updates."""
    state_file = get_appdata_dir() / "installation_state.json"
    current_state = check_installation_state()
    updated_state = {**current_state, **state_updates}
    with open(state_file, "w") as f:
        json.dump(updated_state, f)


def check_vnc_installation() -> bool:
    """Check if VNC dependencies are installed."""
    # Required executables
    required = ["Xvnc"]

    # Window managers (need at least one)
    wm_options = ["matchbox-window-manager", "fluxbox"]

    # Check for required executables
    missing = []
    for cmd in required:
        if not find_executable(cmd):
            missing.append(cmd)

    # Check for at least one window manager
    has_wm = False
    for wm in wm_options:
        if find_executable(wm):
            has_wm = True
            break
    if not has_wm:
        missing.extend(wm_options)

    # Check for noVNC installation
    novnc_paths = ["/usr/share/novnc", "/usr/local/share/novnc", str(Path.home() / "noVNC")]
    has_novnc = any(Path(p).exists() for p in novnc_paths)
    if not has_novnc:
        missing.append("noVNC")

    # Check for websockify (required by noVNC)
    if not find_executable("websockify"):
        missing.append("websockify")

    return len(missing) == 0


def install_vnc_dependencies() -> bool:
    """Install VNC and related dependencies."""
    system = platform.system().lower()
    logger.info("Installing VNC dependencies...")

    try:
        if system == "darwin":
            # macOS: Use Homebrew
            try:
                subprocess.run(["brew", "install", "tigervnc", "fluxbox", "websockify"], check=True)
                # Clone noVNC
                novnc_dir = Path.home() / "noVNC"
                if not novnc_dir.exists():
                    subprocess.run(["git", "clone", "https://github.com/novnc/noVNC.git", str(novnc_dir)], check=True)
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install VNC dependencies via Homebrew: {e}")
                return False

        elif system == "linux":
            try:
                # Try apt-get first (Ubuntu/Debian)
                try:
                    subprocess.run(["sudo", "apt-get", "update"], check=True)
                    subprocess.run(
                        [
                            "sudo",
                            "apt-get",
                            "install",
                            "-y",
                            "tigervnc-standalone-server",
                            "matchbox-window-manager",
                            "fluxbox",
                            "websockify",
                            "git",
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError:
                    # If apt-get fails, try dnf (Fedora)
                    subprocess.run(
                        [
                            "sudo",
                            "dnf",
                            "install",
                            "-y",
                            "tigervnc-server",
                            "matchbox-window-manager",
                            "fluxbox",
                            "websockify",
                            "git",
                        ],
                        check=True,
                    )

                # Clone noVNC
                novnc_dir = Path.home() / "noVNC"
                if not novnc_dir.exists():
                    subprocess.run(["git", "clone", "https://github.com/novnc/noVNC.git", str(novnc_dir)], check=True)
                return True

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install VNC dependencies: {e}")
                logger.warning("""
Failed to install VNC dependencies automatically. Please install manually:

Ubuntu/Debian:
    sudo apt-get update && sudo apt-get install -y tigervnc-standalone-server matchbox-window-manager fluxbox websockify git

Fedora:
    sudo dnf install -y tigervnc-server matchbox-window-manager fluxbox websockify git

Then clone noVNC:
    git clone https://github.com/novnc/noVNC.git ~/noVNC
""")
                return False

        else:
            logger.error(f"Unsupported OS for automatic VNC installation: {system}")
            return False

    except Exception as e:
        logger.error(f"Error during VNC dependency installation: {e}")
        return False


def install_browsers():
    """Install required browsers and dependencies for Playwright."""
    import platform
    from pathlib import Path

    try:
        # Get system-specific Playwright executable path
        system = platform.system().lower()
        if system == "linux":
            executable_path = (
                Path.home() / ".cache/ms-playwright/chromium_headless_shell-1169/chrome-linux/headless_shell"
            )
        elif system == "darwin":
            executable_path = (
                Path.home() / "Library/Caches/ms-playwright/chromium_headless_shell-1169/chrome-mac/headless_shell"
            )
        elif system == "windows":
            executable_path = (
                Path.home() / "AppData/Local/ms-playwright/chromium_headless_shell-1169/chrome-win/headless_shell.exe"
            )
        else:
            executable_path = None

        # Only install if executable doesn't exist
        if not executable_path or not executable_path.exists():
            logger.info("Installing Playwright browsers...")
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            logger.info("Successfully installed Playwright browsers")

        # Only install system dependencies on Linux if not already installed
        state = check_installation_state()
        if system == "linux" and not state["browsers_installed"]:
            try:
                # Try using playwright install-deps first
                subprocess.run([sys.executable, "-m", "playwright", "install-deps"], check=True)
            except subprocess.CalledProcessError:
                # If that fails, try apt-get directly
                try:
                    logger.info("Installing system dependencies...")
                    subprocess.run(["sudo", "apt-get", "update"], check=True)
                    subprocess.run(
                        [
                            "sudo",
                            "apt-get",
                            "install",
                            "-y",
                            "libnss3",
                            "libnspr4",
                            "libasound2",
                            "libatk1.0-0",
                            "libc6",
                            "libcairo2",
                            "libcups2",
                            "libdbus-1-3",
                            "libexpat1",
                            "libfontconfig1",
                            "libgcc1",
                            "libglib2.0-0",
                            "libgtk-3-0",
                            "libpango-1.0-0",
                            "libx11-6",
                            "libx11-xcb1",
                            "libxcb1",
                            "libxcomposite1",
                            "libxcursor1",
                            "libxdamage1",
                            "libxext6",
                            "libxfixes3",
                            "libxi6",
                            "libxrandr2",
                            "libxrender1",
                            "libxss1",
                            "libxtst6",
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error installing system dependencies: {e}")
                    logger.warning("Please run 'sudo apt-get install libnss3 libnspr4 libasound2' manually")
                    return

        # Save successful installation state
        save_installation_state({"browsers_installed": True})

    except Exception as e:
        logger.error(f"Error installing browsers: {e}")
        logger.warning("Please run 'python -m playwright install chromium' manually")
