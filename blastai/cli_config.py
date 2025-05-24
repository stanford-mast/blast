"""Configuration management for BLAST CLI."""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Union, Tuple
from dotenv import load_dotenv

from .models import is_openai_model
from .engine import Engine
from .cli_installation import check_installation_state, install_browsers, check_docker_installation

def is_valid_openai_key(api_key: str) -> bool:
    """Check if a string is a valid OpenAI API key format.
    
    Args:
        api_key: String to check
        
    Returns:
        True if valid format, False otherwise
    """
    return api_key.startswith("sk-") and len(api_key) > 40

def save_api_key(key: str, value: str, env_path: Path) -> bool:
    """Save API key to .env file.
    
    Args:
        key: Environment variable name
        value: API key value
        env_path: Path to .env file
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        if not env_path.exists():
            env_path.write_text(f"{key}={value}\n")
        else:
            content = env_path.read_text()
            if f"{key}=" in content:
                lines = content.splitlines()
                new_lines = []
                for line in lines:
                    if line.startswith(f"{key}="):
                        new_lines.append(f"{key}={value}")
                    else:
                        new_lines.append(line)
                env_path.write_text("\n".join(new_lines) + "\n")
            else:
                with env_path.open("a") as f:
                    f.write(f"\n{key}={value}\n")
        return True
    except Exception as e:
        print(f"\nError saving API key: {e}")
        return False

def check_model_api_key(model_name: str, env_path: Optional[Path] = None) -> bool:
    """Check if required API key is available for the given model.
    
    Args:
        model_name: Name of the model to check API key for
        env_path: Optional path to .env file for saving API key
        
    Returns:
        True if API key is available, False otherwise
    """
    # Check if model requires OpenAI API key
    if is_openai_model(model_name):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return True
            
        print(f"OpenAI API key required for {model_name}. [https://platform.openai.com/api-keys]")
        
        while True:
            api_key = input("Enter your OpenAI API key: ").strip()
            if is_valid_openai_key(api_key):
                # Save to .env file if path provided
                if env_path and save_api_key("OPENAI_API_KEY", api_key, env_path):
                    print("\nAPI key saved successfully!")
                
                os.environ["OPENAI_API_KEY"] = api_key
                return True
            else:
                print("\nInvalid API key format. API keys should start with 'sk-' and be at least 40 characters long.")
                retry = input("Would you like to try again? (y/n): ").lower()
                if retry != 'y':
                    return False
    
    # For other models, no API key check needed yet
    return True

def parse_env_param(env_param: Optional[str]) -> Dict[str, str]:
    """Parse --env parameter value into a dictionary.
    
    Args:
        env_param: String in format "KEY1=value1,KEY2=value2"
        
    Returns:
        Dictionary of environment variable overrides
    """
    if not env_param:
        return {}
        
    env_dict = {}
    for pair in env_param.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        env_dict[key.strip()] = value.strip()
    return env_dict

def load_environment(env: Optional[Union[str, Dict[str, str]]] = None) -> Path:
    """Load environment variables in priority order.
    
    Args:
        env: Optional environment variables from --env parameter
        
    Returns:
        Path to .env file for saving new variables
    """
    # 1. Load CLI args (highest priority)
    if isinstance(env, str):
        env_vars = parse_env_param(env)
        if env_vars:
            for key, value in env_vars.items():
                os.environ[key] = value
    elif isinstance(env, dict):
        for key, value in env.items():
            os.environ[key] = value
            
    # 2. Load .env file (lower priority)
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        
    return env_path

async def setup_environment(env: Optional[str] = None, config_path: Optional[str] = None) -> Tuple[Path, Dict]:
    """Set up environment variables and load config.
    
    Args:
        env: Optional environment variables from --env parameter
        config_path: Optional path to config file
        
    Returns:
        Tuple of (env_path, config)
    """
    # Load environment variables first
    env_path = load_environment(env)
    
    # Load config
    config = Engine.load_config(config_path)
    
    # Check Steel requirements if enabled
    if config['constraints'].get('require_steel', False):
        requirements_ok, error_msg = Engine.check_requirements(config_path=config_path)
        if not requirements_ok:
            if "Docker" in error_msg:
                # Just print instructions and exit - check_docker_installation handles the output
                _, _ = check_docker_installation()
                sys.exit(1)
            else:
                print(f"\nError: {error_msg}")
                sys.exit(1)
    
    # Check for required API keys based on config
    if not check_model_api_key(config['constraints']['llm_model'], env_path):
        print("\nRequired API key not found. Exiting.")
        sys.exit(1)
        
    if config['constraints'].get('llm_model_mini') and not check_model_api_key(config['constraints']['llm_model_mini'], env_path):
        print("\nRequired API key not found for mini model. Exiting.")
        sys.exit(1)
    
    # Check if already installed
    from .cli_installation import check_installation_state, install_browsers
    already_installed = check_installation_state()
    
    # Install browsers and dependencies if needed
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            if not p.chromium.executable_path.exists():
                install_browsers(quiet=already_installed)
    except Exception:
        install_browsers(quiet=already_installed)
        
    return env_path, config