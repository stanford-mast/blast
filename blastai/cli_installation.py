"""Installation and dependency management for BLAST CLI."""

import os
import sys
import shutil
import subprocess
import json
import logging
from pathlib import Path
from typing import Optional

from .utils import get_appdata_dir

logger = logging.getLogger('web')

def find_executable(*names: str) -> Optional[str]:
    """Find the first available executable from the given names."""
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None

def check_node_installation() -> Optional[str]:
    """Check if Node.js is installed and available."""
    node_cmd = find_executable('node', 'node.exe')
    if not node_cmd:
        logger.error("Node.js not found")
        # Show installation instructions (essential user guidance)
        print("""
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
    """Check if npm is installed and available."""
    npm_cmd = find_executable('npm', 'npm.cmd')
    if not npm_cmd:
        logger.error("npm not found")
        # Show installation instructions (essential user guidance)
        instructions = "\nError: npm not found. The web frontend requires Node.js/npm.\n\nTo install Node.js and npm:\n"
        
        if sys.platform == 'win32':
            instructions += """
On Windows:
1. Visit https://nodejs.org
2. Download and run the Windows Installer (.msi)
3. Follow the installation wizard (this will install both Node.js and npm)
4. Open a new terminal and run 'npm --version' to verify
"""
        elif sys.platform == 'darwin':
            instructions += """
On macOS:
Option 1 - Using Homebrew (recommended):
1. Install Homebrew if not installed:
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
2. Install Node.js (includes npm):
   brew install node

Option 2 - Manual installation:
1. Visit https://nodejs.org
2. Download and run the macOS Installer (.pkg)
3. Open a new terminal and run 'npm --version' to verify
"""
        else:  # Linux
            instructions += """
On Linux:
Option 1 - Using package manager (recommended):
Ubuntu/Debian:
1. sudo apt update
2. sudo apt install nodejs npm

Fedora:
sudo dnf install nodejs npm

Option 2 - Using Node Version Manager (nvm):
1. Install nvm:
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
2. Restart your terminal
3. Install Node.js (includes npm):
   nvm install --lts
"""
        instructions += """
After installation:
1. Close and reopen your terminal
2. Run 'npm --version' to verify npm is installed
3. Run 'blastai serve' again to start the web frontend
"""
        print(instructions)
        return None
        
    return npm_cmd

def check_installation_state() -> bool:
    """Check if browsers and dependencies are already installed."""
    state_file = get_appdata_dir() / "installation_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            return state.get("browsers_installed", False)
    return False

def save_installation_state():
    """Save that installation was successful."""
    state_file = get_appdata_dir() / "installation_state.json"
    with open(state_file, "w") as f:
        json.dump({"browsers_installed": True}, f)

def install_browsers():
    """Install required browsers and dependencies for Playwright."""
    import platform
    from pathlib import Path
    
    try:
        # Get system-specific Playwright executable path
        system = platform.system().lower()
        if system == 'linux':
            executable_path = Path.home() / '.cache/ms-playwright/chromium_headless_shell-1169/chrome-linux/headless_shell'
        elif system == 'darwin':
            executable_path = Path.home() / 'Library/Caches/ms-playwright/chromium_headless_shell-1169/chrome-mac/headless_shell'
        elif system == 'windows':
            executable_path = Path.home() / 'AppData/Local/ms-playwright/chromium_headless_shell-1169/chrome-win/headless_shell.exe'
        else:
            executable_path = None
            
        # Only install if executable doesn't exist
        if not executable_path or not executable_path.exists():
            logger.info("Installing Playwright browsers...")
            subprocess.run([sys.executable, '-m', 'playwright', 'install', 'chromium'], check=True)
            logger.info("Successfully installed Playwright browsers")
        
        # Only install system dependencies on Linux if not already installed
        if system == 'linux' and not check_installation_state():
            try:
                # Try using playwright install-deps first
                subprocess.run([sys.executable, '-m', 'playwright', 'install-deps'], check=True)
            except subprocess.CalledProcessError:
                # If that fails, try apt-get directly
                try:
                    logger.info("Installing system dependencies...")
                    subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                    subprocess.run(['sudo', 'apt-get', 'install', '-y',
                        'libnss3', 'libnspr4', 'libasound2', 'libatk1.0-0',
                        'libc6', 'libcairo2', 'libcups2', 'libdbus-1-3',
                        'libexpat1', 'libfontconfig1', 'libgcc1', 'libglib2.0-0',
                        'libgtk-3-0', 'libpango-1.0-0', 'libx11-6', 'libx11-xcb1',
                        'libxcb1', 'libxcomposite1', 'libxcursor1', 'libxdamage1',
                        'libxext6', 'libxfixes3', 'libxi6', 'libxrandr2',
                        'libxrender1', 'libxss1', 'libxtst6'
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error installing system dependencies: {e}")
                    print("Please run 'sudo apt-get install libnss3 libnspr4 libasound2' manually")
                    return
                
        # Save successful installation state
        save_installation_state()
        
    except Exception as e:
        logger.error(f"Error installing browsers: {e}")
        print("Please run 'python -m playwright install chromium' manually")