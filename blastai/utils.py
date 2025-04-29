"""Utility functions for BlastAI."""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import json
from typing import Dict, Optional, List

def estimate_llm_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: Optional[int] = 0
) -> float:
    """Estimate LLM cost based on token counts and model pricing.
    
    Args:
        model_name: Name of the LLM model
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (default: 0)
        
    Returns:
        Estimated cost in USD
    """
    # Load pricing config
    pricing_path = os.path.join(os.path.dirname(__file__), 'pricing_openai_api.json')
    with open(pricing_path) as f:
        pricing_config = json.load(f)
        
    if model_name not in pricing_config["models"]:
        return 0.0
        
    pricing = pricing_config["models"][model_name]
    
    # Calculate costs
    input_cost = (prompt_tokens - cached_tokens) * pricing["input"] / 1_000_000
    cached_cost = cached_tokens * pricing.get("cachedInput", pricing["input"]) / 1_000_000
    output_cost = completion_tokens * pricing["output"] / 1_000_000
    
    return input_cost + cached_cost + output_cost

def get_appdata_dir() -> Path:
    """Get the appropriate app data directory for the current platform.
    
    Returns:
        Path to the BlastAI app data directory
    """
    if sys.platform == "win32":
        # Windows: %LOCALAPPDATA%\blastai
        base_dir = os.environ.get("LOCALAPPDATA")
        if not base_dir:
            base_dir = os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        # macOS: ~/Library/Application Support/blastai
        base_dir = os.path.expanduser("~/Library/Application Support")
    else:
        # Linux/Unix: ~/.local/share/blastai
        base_dir = os.path.expanduser("~/.local/share")
        
    app_dir = Path(base_dir) / "blastai"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir

def find_local_browser(cached_path: Optional[str] = None) -> Optional[str]:
    """Find local Chrome/Chromium browser installation.
    
    Args:
        cached_path: Optional previously found browser path to check first
        
    Returns:
        Path to browser binary if found, None otherwise
    """
    # First check cached path if provided
    if cached_path and cached_path != "auto":
        if os.path.exists(cached_path):
            return cached_path
        return None

    paths: List[str] = []
    
    if sys.platform == "win32":
        # Windows paths
        program_files = [
            os.environ.get("ProgramFiles", "C:\\Program Files"),
            os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        ]
        
        # Program Files locations
        for pf in program_files:
            paths.extend([
                os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(pf, "Google", "Chrome Beta", "Application", "chrome.exe"),
                os.path.join(pf, "Google", "Chrome Dev", "Application", "chrome.exe"),
                os.path.join(pf, "Google", "Chrome SxS", "Application", "chrome.exe"),
                os.path.join(pf, "Chromium", "Application", "chrome.exe")
            ])
            
        # Local AppData locations
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            paths.extend([
                os.path.join(local_app_data, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(local_app_data, "Google", "Chrome Beta", "Application", "chrome.exe"),
                os.path.join(local_app_data, "Google", "Chrome Dev", "Application", "chrome.exe"),
                os.path.join(local_app_data, "Google", "Chrome SxS", "Application", "chrome.exe"),
                os.path.join(local_app_data, "Chromium", "Application", "chrome.exe")
            ])
            
        # Check registry on Windows
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe") as key:
                chrome_path = winreg.QueryValue(key, None)
                if chrome_path:
                    paths.append(chrome_path)
        except (ImportError, FileNotFoundError, OSError):
            pass
            
    elif sys.platform == "darwin":
        # macOS paths
        paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
            "/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev",
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            os.path.expanduser("~/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"),
            os.path.expanduser("~/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev"),
            os.path.expanduser("~/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"),
            os.path.expanduser("~/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
        
        # Check Homebrew locations
        brew_paths = [
            "/usr/local/Caskroom/google-chrome/latest/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/usr/local/Caskroom/google-chrome-beta/latest/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
            "/usr/local/Caskroom/google-chrome-dev/latest/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev",
            "/usr/local/Caskroom/google-chrome-canary/latest/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            "/opt/homebrew/Caskroom/google-chrome/latest/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/opt/homebrew/Caskroom/google-chrome-beta/latest/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
            "/opt/homebrew/Caskroom/google-chrome-dev/latest/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev",
            "/opt/homebrew/Caskroom/google-chrome-canary/latest/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"
        ]
        paths.extend(brew_paths)
        
    else:
        # Linux paths
        paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/google-chrome-beta",
            "/usr/bin/google-chrome-unstable",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/opt/google/chrome/google-chrome",
            "/opt/google/chrome/chrome",
            "/opt/chromium/chrome",
            os.path.expanduser("~/.local/bin/google-chrome"),
            os.path.expanduser("~/.local/bin/chromium")
        ]
        
        # For WSL, check Windows paths
        if "microsoft" in os.uname().release.lower():
            win_paths = [
                "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
                "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                "/mnt/c/Program Files/Google/Chrome Beta/Application/chrome.exe",
                "/mnt/c/Program Files (x86)/Google/Chrome Beta/Application/chrome.exe",
                "/mnt/c/Program Files/Google/Chrome Dev/Application/chrome.exe",
                "/mnt/c/Program Files (x86)/Google/Chrome Dev/Application/chrome.exe",
                "/mnt/c/Program Files/Google/Chrome SxS/Application/chrome.exe",
                "/mnt/c/Program Files (x86)/Google/Chrome SxS/Application/chrome.exe"
            ]
            paths.extend(win_paths)
            
            # Also check Windows user's AppData
            try:
                windows_username = subprocess.check_output(["cmd.exe", "/c", "echo %USERNAME%"], text=True).strip()
                if windows_username:
                    appdata_paths = [
                        f"/mnt/c/Users/{windows_username}/AppData/Local/Google/Chrome/Application/chrome.exe",
                        f"/mnt/c/Users/{windows_username}/AppData/Local/Google/Chrome Beta/Application/chrome.exe",
                        f"/mnt/c/Users/{windows_username}/AppData/Local/Google/Chrome Dev/Application/chrome.exe",
                        f"/mnt/c/Users/{windows_username}/AppData/Local/Google/Chrome SxS/Application/chrome.exe"
                    ]
                    paths.extend(appdata_paths)
            except subprocess.CalledProcessError:
                pass

    # Check PATH
    path_browsers = ["google-chrome", "google-chrome-stable", "google-chrome-beta",
                    "google-chrome-unstable", "chromium", "chromium-browser"]
    for browser in path_browsers:
        path = shutil.which(browser)
        if path:
            paths.append(path)

    # Return first existing path
    for path in paths:
        if os.path.exists(path):
            return path
            
    return None