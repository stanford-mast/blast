"""Installation and dependency management for BLAST CLI."""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple

import docker
from .utils import get_appdata_dir

def find_executable(*names: str) -> Optional[str]:
    """Find the first available executable from the given names."""
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None

def is_wsl() -> bool:
    """Check if running under Windows Subsystem for Linux."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False

def check_docker_installation() -> Tuple[bool, str]:
    """Check if Docker is installed and running.
    
    Returns:
        Tuple of (is_available, error_message)
        is_available: True if Docker is installed and running
        error_message: Error message if Docker is not available, empty string otherwise
    """
    try:
        client = docker.from_env()
        client.ping()
        return True, ""
    except docker.errors.DockerException as e:
        error_msg = str(e)
        # Check for permission error in the full error chain
        if "Permission denied" in error_msg or "PermissionError(13" in error_msg:
            print("The current BLAST configuration requires Steel for browser management.\n")
            print("You can either:")
            print("1. Set up Docker to use Steel (recommended)")
            print("2. Set require_steel=false in blastai/default_config.yaml")
            print("\nDocker is installed but you don't have permission to use it.")
            print("To fix this:")
            print("1. Run: sudo usermod -aG docker $USER")
            print("2. Run: newgrp docker")
            print("3. Try 'blastai serve' again")
            return False, "Permission denied accessing Docker socket"
        elif is_wsl() and ("not found" in error_msg or "Connection aborted" in error_msg):
            print("The current BLAST configuration requires Steel for browser management.")
            print()
            print("You can either:")
            print("1. Install Docker to use Steel (recommended)")
            print("2. Set require_steel=false in blastai/default_config.yaml")
            print("\nTo set up Docker in WSL 2:")
            print("1. Install Docker Desktop for Windows")
            print("   Visit: https://docs.docker.com/desktop/install/windows-install/")
            print("2. In Docker Desktop settings:")
            print("   - Go to Settings > Resources > WSL Integration")
            print("   - Enable 'Ubuntu' or your WSL distro")
            print("3. Restart Docker Desktop")
            print("4. Open a new WSL terminal")
            print("\nFor detailed instructions, visit:")
            print("https://docs.docker.com/go/wsl2/")
            print("\nAfter installation:")
            print("1. Open Docker Desktop")
            print("2. Wait for Docker Desktop to finish starting")
            print("3. Try 'blastai serve' again")
            return False, "Docker is not available in WSL. Docker Desktop integration needs to be enabled."
        elif "ConnectionError" in error_msg:
            print("The current BLAST configuration requires Steel for browser management.")
            print("You can either:")
            print("1. Start Docker to use Steel (recommended)")
            print("2. Set require_steel=false in blastai/default_config.yaml")
            print("\nDocker daemon is not running.")
            print("Please start Docker and try again.")
            return False, "Docker daemon is not running"
        elif "not found" in error_msg:
            print("The current BLAST configuration requires Steel for browser management.")
            print("You can either:")
            print("1. Install Docker to use Steel (recommended)")
            print("2. Set require_steel=false in blastai/default_config.yaml")
            print("\nTo install Docker:")
            
            if sys.platform == 'win32':
                print("\nOn Windows:")
                print("1. Visit https://docs.docker.com/desktop/install/windows-install/")
                print("2. Download and run Docker Desktop Installer")
                print("3. Follow the installation wizard")
                print("4. Start Docker Desktop")
                
            elif sys.platform == 'darwin':
                print("\nOn macOS:")
                print("1. Visit https://docs.docker.com/desktop/install/mac-install/")
                print("2. Download and install Docker Desktop")
                print("3. Start Docker Desktop")
                
            else:  # Linux
                print("\nOn Linux:")
                print("Ubuntu/Debian:")
                print("1. sudo apt-get update")
                print("2. sudo apt-get install docker.io")
                print("3. sudo systemctl start docker")
                print("4. sudo systemctl enable docker")
                print("\nFedora:")
                print("1. sudo dnf install docker")
                print("2. sudo systemctl start docker")
                print("3. sudo systemctl enable docker")
            
            return False, "Docker is not installed"
        else:
            return False, f"Docker error: {e}"
    except Exception as e:
        return False, f"Failed to connect to Docker: {e}"

def check_node_installation() -> Optional[str]:
    """Check if Node.js is installed and available."""
    node_cmd = find_executable('node', 'node.exe')
    if not node_cmd:
        print("\nNode.js not found. The web frontend requires Node.js.")
        print("\nTo install Node.js:")
        print("1. Visit https://nodejs.org")
        print("2. Download and install the LTS version")
        print("3. Run 'node --version' to verify installation")
        print("\nFalling back to CLI frontend for now...")
        print("Once Node.js is installed, run 'blastai serve' again to use the web frontend\n")
        return None
    return node_cmd

def check_npm_installation() -> Optional[str]:
    """Check if npm is installed and available."""
    npm_cmd = find_executable('npm', 'npm.cmd')
    if not npm_cmd:
        print("\nError: npm not found. The web frontend requires Node.js/npm.")
        print("\nTo install Node.js and npm:")
        
        if sys.platform == 'win32':
            print("\nOn Windows:")
            print("1. Visit https://nodejs.org")
            print("2. Download and run the Windows Installer (.msi)")
            print("3. Follow the installation wizard (this will install both Node.js and npm)")
            print("4. Open a new terminal and run 'npm --version' to verify")
            
        elif sys.platform == 'darwin':
            print("\nOn macOS:")
            print("Option 1 - Using Homebrew (recommended):")
            print("1. Install Homebrew if not installed:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            print("2. Install Node.js (includes npm):")
            print("   brew install node")
            print("\nOption 2 - Manual installation:")
            print("1. Visit https://nodejs.org")
            print("2. Download and run the macOS Installer (.pkg)")
            print("3. Open a new terminal and run 'npm --version' to verify")
            
        else:  # Linux
            print("\nOn Linux:")
            print("Option 1 - Using package manager (recommended):")
            print("Ubuntu/Debian:")
            print("1. sudo apt update")
            print("2. sudo apt install nodejs npm")
            print("\nFedora:")
            print("sudo dnf install nodejs npm")
            print("\nOption 2 - Using Node Version Manager (nvm):")
            print("1. Install nvm:")
            print("   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash")
            print("2. Restart your terminal")
            print("3. Install Node.js (includes npm):")
            print("   nvm install --lts")
            
        print("\nAfter installation:")
        print("1. Close and reopen your terminal")
        print("2. Run 'npm --version' to verify npm is installed")
        print("3. Run 'blastai serve' again to start the web frontend")
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

def install_browsers(quiet: bool = False):
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
            if not quiet:
                print("Installing Playwright browsers...")
            subprocess.run([sys.executable, '-m', 'playwright', 'install', 'chromium'], check=True)
            if not quiet:
                print("Successfully installed Playwright browsers")
        
        # Only install system dependencies on Linux if not already installed
        if system == 'linux' and not check_installation_state():
            try:
                # Try using playwright install-deps first
                subprocess.run([sys.executable, '-m', 'playwright', 'install-deps'], check=True)
            except subprocess.CalledProcessError:
                # If that fails, try apt-get directly
                try:
                    print("Installing dependencies...")
                    subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                    subprocess.run(['sudo', 'apt-get', 'install', '-y',
                        'libnss3',
                        'libnspr4',
                        'libasound2',
                        'libatk1.0-0',
                        'libc6',
                        'libcairo2',
                        'libcups2',
                        'libdbus-1-3',
                        'libexpat1',
                        'libfontconfig1',
                        'libgcc1',
                        'libglib2.0-0',
                        'libgtk-3-0',
                        'libpango-1.0-0',
                        'libx11-6',
                        'libx11-xcb1',
                        'libxcb1',
                        'libxcomposite1',
                        'libxcursor1',
                        'libxdamage1',
                        'libxext6',
                        'libxfixes3',
                        'libxi6',
                        'libxrandr2',
                        'libxrender1',
                        'libxss1',
                        'libxtst6'
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error installing system dependencies: {e}")
                    print("Please run 'sudo apt-get install libnss3 libnspr4 libasound2' manually")
                    return
                
        # Save successful installation state
        save_installation_state()
        
    except Exception as e:
        print(f"Error installing browsers: {e}")
        print("Please run 'python -m playwright install chromium' manually")