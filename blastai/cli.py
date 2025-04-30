# Set anonymized telemetry to false before any imports
import os

from dotenv import load_dotenv
os.environ["ANONYMIZED_TELEMETRY"] = "false"

"""CLI interface for BLAST."""

import sys
import click
import httpx
import uvicorn
import asyncio
import threading
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from openai import OpenAI

from .server import app, init_app_state
from .logging_setup import should_show_metrics, setup_logging
from .config import Settings

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

async def run_cli_frontend(server_port: int):
    """Run CLI frontend for interacting with BLAST."""
    client = OpenAI(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{server_port}"
    )
    
    previous_response_id = None
    
    while True:
        try:
            task = input("> ")
            if task.lower() == 'exit':
                break
                
            stream = client.responses.create(
                model="not-needed",
                input=task,
                stream=True,
                previous_response_id=previous_response_id
            )
            
            # Track the current thought
            current_thought = ""
            
            for event in stream:
                if event.type == "response.completed":
                    previous_response_id = event.response.id
                elif event.type == "response.output_text.delta":
                    # Accumulate the thought
                    if ' ' in event.delta:  # Skip screenshots
                        current_thought += event.delta
                elif event.type == "response.output_text.done":
                    # Print complete thought and reset
                    if current_thought:
                        print(current_thought)
                        current_thought = ""
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

async def run_cli_server(server, port: int):
    """Run server with CLI frontend."""
    server.should_exit = lambda: False
    server.startup_complete = lambda: (
        asyncio.get_event_loop().create_task(run_cli_frontend(port))
    )
    await server.serve()

@click.group()
def cli():
    """BLAST CLI tool for browser automation."""
    pass

def find_available_port(start_port: int, max_attempts: int = 10) -> Optional[int]:
    """Find an available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

@cli.command()
@click.option('--config', type=str, help='Path to config YAML file')
@click.option('--no-metrics-output', is_flag=True, help='Disable metrics output')
@click.option('--server-port', type=int, default=8000, help='Port for the backend server')
@click.option('--web-port', type=int, default=3000, help='Port for the web frontend')
@click.argument('component', type=click.Choice(['web', 'cli', 'engine']), required=False)
def serve(config: Optional[str], no_metrics_output: bool, server_port: int, web_port: int, component: Optional[str] = None):
    """Start BLAST components."""
    # Initialize app state with config (this loads default_config.yaml)
    init_app_state(config)
    
    async def run_web_frontend():
        """Run just the web frontend."""
        frontend_dir = Path(__file__).parent / 'frontend'
        
        # Check Node.js and npm installation
        if not check_node_installation():
            return
            
        npm_cmd = check_npm_installation()
        if not npm_cmd:
            return
            
        # Install dependencies if needed
        if not (frontend_dir / 'node_modules').exists():
            print("Installing frontend dependencies...")
            try:
                subprocess.run([npm_cmd, 'install'], cwd=frontend_dir, check=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Error installing frontend dependencies: {e}")
                return

        # Start frontend process
        try:
            process = subprocess.Popen(
                [npm_cmd, 'run', 'dev'],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error starting frontend: {e}")
            return

        # Monitor frontend output
        def print_output():
            try:
                for line in process.stdout:
                    # Show frontend output for debugging
                    print(f"Web: {line.strip()}")
            except (ValueError, IOError) as e:
                print(f"Error reading frontend output: {e}")
        
        output_thread = threading.Thread(target=print_output, daemon=True)
        output_thread.start()

        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("\nShutting down web frontend...")
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    async def run_standalone_cli(actual_server_port):
        """Run just the CLI frontend."""
        client = OpenAI(
            api_key="not-needed",
            base_url=f"http://127.0.0.1:{actual_server_port}"
        )
        
        previous_response_id = None
        
        while True:
            try:
                task = input("> ")
                if task.lower() == 'exit':
                    break
                    
                stream = None
                try:
                    stream = client.responses.create(
                        model="not-needed",
                        input=task,
                        stream=True,
                        previous_response_id=previous_response_id
                    )
                    
                    # Track the current thought
                    current_thought = ""
                    final_result = None
                    
                    for event in stream:
                        if event.type == "response.completed":
                            previous_response_id = event.response.id
                            final_result = event.response.output[0].content[0].text
                        elif event.type == "response.output_text.delta":
                            # Accumulate the thought
                            if ' ' in event.delta:  # Skip screenshots
                                current_thought += event.delta
                        elif event.type == "response.output_text.done":
                            # Print complete thought and reset
                            if current_thought:
                                print(current_thought)
                                current_thought = ""
                    
                    # Print final result if different from last thought
                    if final_result and (not current_thought or final_result != current_thought):
                        print(final_result, flush=True)
                    
                except Exception as e:
                    if "Backend server not running" in str(e):
                        print("Error: Backend server not running. Start it with 'blastai serve engine'")
                    else:
                        # Ignore stream termination errors
                        pass
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: Backend server not running. Start it with 'blastai serve engine'")
                continue

    async def display_metrics(client, settings: Settings, server_port: int):
        """Display and update metrics every 5s."""
        # Constants for metrics display
        METRICS_LINES = 10  # Including blank line at start
        BLANK_METRICS = {
            'tasks': {'scheduled': 0, 'running': 0, 'completed': 0},
            'concurrent_browsers': 0,
            'memory_usage_gb': 0.0,
            'total_cost': 0.00
        }
        
        def print_metrics(metrics=None):
            """Print metrics in a consistent format."""
            if metrics is None:
                metrics = BLANK_METRICS
            
            # Move cursor up if needed (not on first print)
            if print_metrics.initialized:
                print(f"\033[{METRICS_LINES}A", end='')
            print_metrics.initialized = True
            
            # Clear lines and print metrics
            print("\033[J", end='')  # Clear everything below
            print()  # Blank line for spacing
            print("Tasks:")
            print(f"  Scheduled: {metrics['tasks']['scheduled']}")
            print(f"  Running:   {metrics['tasks']['running']}")
            print(f"  Completed: {metrics['tasks']['completed']}")
            print()
            print("Resources:")
            print(f"  Active browsers: {metrics['concurrent_browsers']}")
            print(f"  Memory usage:    {metrics['memory_usage_gb']:.1f} GB")
            print(f"  Total cost:      ${metrics['total_cost']:.4f}", flush=True)
        
        # Initialize the print_metrics function state
        print_metrics.initialized = False
        
        # Print initial metrics
        print_metrics()
        
        while True:
            try:
                # Get metrics from server
                response = await client.get(f"http://127.0.0.1:{server_port}/metrics")
                metrics = response.json()
                print_metrics(metrics)
            except Exception:
                print_metrics()
            await asyncio.sleep(5)

    async def run_server_and_frontend():
        """Run server and frontend concurrently."""
        # Initialize app state with config (this loads default_config.yaml)
        init_app_state(config)
        
        # Get settings that were loaded by init_app_state
        from .server import _settings
        if _settings is None:
            raise RuntimeError("Settings not initialized properly")
        settings = _settings

        # Find available ports if the requested ones are in use
        actual_server_port = find_available_port(server_port)
        if not actual_server_port:
            print(f"\nError: Could not find available port for server (tried ports {server_port}-{server_port+9})")
            return
        
        actual_web_port = find_available_port(web_port)
        if not actual_web_port:
            print(f"\nError: Could not find available port for web frontend (tried ports {web_port}-{web_port+9})")
            return

        # Create server with proper logging level
        server_config = uvicorn.Config(
            "blastai.server:app",
            host="127.0.0.1",
            port=actual_server_port,
            log_level=settings.blastai_log_level.lower(),
            reload=False,
            workers=1,
            lifespan="on",
            timeout_keep_alive=5,
            timeout_graceful_shutdown=10,
            access_log=False
        )
        server = uvicorn.Server(server_config)
        server.force_exit = False  # Allow graceful shutdown

        # Print server endpoint
        if component == 'engine':
            print(f"Server: http://127.0.0.1:{actual_server_port}")
        elif component == 'web':
            print(f"Web: http://localhost:{actual_web_port}")
        elif component is None:
            print(f"Server: http://127.0.0.1:{actual_server_port}")
            print(f"Web: http://localhost:{actual_web_port}")

        # Create metrics task if needed
        metrics_task = None
        metrics_client = None
        try:
            # Only show metrics if:
            # 1. Metrics output is not disabled via --no-metrics-output
            # 2. Running engine component (either standalone or with web)
            # 3. Both log levels are ERROR or CRITICAL
            if (not no_metrics_output and
                (component == 'engine' or component is None) and
                should_show_metrics(settings)):
                metrics_client = httpx.AsyncClient()
                metrics_task = asyncio.create_task(display_metrics(metrics_client, settings, actual_server_port))

            if component is None:
                # Default behavior: run both backend and web frontend
                frontend_dir = Path(__file__).parent / 'frontend'
                
                # Check Node.js and npm installation
                if not check_node_installation():
                    await run_cli_server(server, actual_server_port)
                    return
                    
                npm_cmd = check_npm_installation()
                if not npm_cmd:
                    await run_cli_server(server, actual_server_port)
                    return
                    
                # Install dependencies if needed
                if not (frontend_dir / 'node_modules').exists():
                    try:
                        subprocess.run([npm_cmd, 'install'], cwd=frontend_dir, check=True, text=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error installing frontend dependencies: {e}")
                        await run_cli_server(server, actual_server_port)
                        return

                # Start frontend process
                try:
                    # Set environment variables for the frontend process
                    frontend_env = os.environ.copy()
                    frontend_env['NEXT_PUBLIC_SERVER_PORT'] = str(actual_server_port)
                    frontend_env['PORT'] = str(actual_web_port)
                    
                    process = subprocess.Popen(
                        [npm_cmd, 'run', 'dev', f'--port={actual_web_port}'],
                        cwd=frontend_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=frontend_env
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error starting frontend: {e}")
                    await run_cli_server(server)
                    return

                # Monitor frontend output
                frontend_ready = threading.Event()
                def print_output():
                    try:
                        for line in process.stdout:
                            line = line.strip()
                            # Only show errors and important info
                            if any(x in line.lower() for x in ['error:', 'warn:', '✓ ready', 'invalid']):
                                print(f"Web: {line}")
                            # Set ready event when frontend is loaded
                            if '✓ ready' in line:
                                frontend_ready.set()
                    except (ValueError, IOError) as e:
                        print(f"Error reading frontend output: {e}")
                output_thread = threading.Thread(target=print_output, daemon=True)
                output_thread.start()

                # Wait for frontend to be ready before showing metrics
                try:
                    frontend_ready.wait(timeout=10)
                except TimeoutError:
                    print("Warning: Frontend startup took longer than expected")

                # Run server until interrupted
                try:
                    await server.serve()
                except asyncio.CancelledError:
                    # Let server handle its own shutdown
                    await server.shutdown()
                    raise
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

            elif component == 'web':
                await run_web_frontend()
            elif component == 'cli':
                await run_standalone_cli(actual_server_port)
            elif component == 'engine':
                # Only run the backend server
                await server.serve()

        except asyncio.CancelledError:
            # Let server handle its own shutdown
            await server.shutdown()
            raise
        finally:
            # Clean up metrics task if it exists
            if metrics_task and not metrics_task.done():
                metrics_task.cancel()
                try:
                    await metrics_task
                except asyncio.CancelledError:
                    pass
            if metrics_client:
                await metrics_client.aclose()
    
    # Run everything in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Create main task
    main_task = None
    server_task = None
    
    try:
        # Run the main coroutine
        main_task = loop.create_task(run_server_and_frontend())
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        if main_task and not main_task.done():
            # Cancel the main task
            main_task.cancel()
            try:
                # Wait for cancellation to complete
                loop.run_until_complete(main_task)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                import traceback
                print(f"\nError: {e}")
                print("\nStack trace:")
                print(traceback.format_exc())
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        try:
            # Clean up any remaining tasks
            tasks = asyncio.all_tasks(loop)
            if tasks:
                # Cancel all tasks
                for task in tasks:
                    task.cancel()
                
                # Wait for tasks to complete with timeout
                cleanup_task = asyncio.gather(*tasks, return_exceptions=True)
                try:
                    loop.run_until_complete(asyncio.wait_for(cleanup_task, timeout=5.0))
                except asyncio.TimeoutError:
                    pass
        finally:
            loop.close()

def check_installation_state() -> bool:
    """Check if browsers and dependencies are already installed."""
    from pathlib import Path
    from .utils import get_appdata_dir
    
    state_file = get_appdata_dir() / "installation_state.json"
    if state_file.exists():
        import json
        with open(state_file) as f:
            state = json.load(f)
            return state.get("browsers_installed", False)
    return False

def save_installation_state():
    """Save that installation was successful."""
    from pathlib import Path
    from .utils import get_appdata_dir
    import json
    
    state_file = get_appdata_dir() / "installation_state.json"
    with open(state_file, "w") as f:
        json.dump({"browsers_installed": True}, f)

def install_browsers(quiet: bool = False):
    """Install required browsers and dependencies for Playwright."""
    import subprocess
    import sys
    import platform
    
    try:
        # Check if already installed
        if check_installation_state():
            return
            
        # First install browsers
        subprocess.run([sys.executable, '-m', 'playwright', 'install', 'chromium'], check=True)
        if not quiet:
            print("Successfully installed Playwright browsers")
        
        # Then install system dependencies if on Linux
        if platform.system() == 'Linux':
            try:
                # Try using playwright install-deps first
                subprocess.run([sys.executable, '-m', 'playwright', 'install-deps'], check=True)
            except subprocess.CalledProcessError:
                # If that fails, try apt-get directly
                try:
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

def check_openai_api_key():
    """Check if OpenAI API key is available and prompt if not found."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return True
        
    # Check .env file in current directory
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return True
    
    print("\nOpenAI API key not found. This is required for BLAST to run.")
    print("You can get an API key from https://platform.openai.com/api-keys")
    
    while True:
        api_key = input("\nPlease enter your OpenAI API key: ").strip()
        if api_key.startswith("sk-") and len(api_key) > 40:
            try:
                # Save to .env file
                if not env_path.exists():
                    env_path.write_text(f"OPENAI_API_KEY={api_key}\n")
                else:
                    content = env_path.read_text()
                    if "OPENAI_API_KEY=" in content:
                        lines = content.splitlines()
                        new_lines = []
                        for line in lines:
                            if line.startswith("OPENAI_API_KEY="):
                                new_lines.append(f"OPENAI_API_KEY={api_key}")
                            else:
                                new_lines.append(line)
                        env_path.write_text("\n".join(new_lines) + "\n")
                    else:
                        with env_path.open("a") as f:
                            f.write(f"\nOPENAI_API_KEY={api_key}\n")
                
                os.environ["OPENAI_API_KEY"] = api_key
                print("\nAPI key saved successfully!")
                return True
            except Exception as e:
                print(f"\nError saving API key: {e}")
                print("Please try again.")
        else:
            print("\nInvalid API key format. API keys should start with 'sk-' and be at least 40 characters long.")
            retry = input("Would you like to try again? (y/n): ").lower()
            if retry != 'y':
                return False

def main():
    """Main entry point for CLI."""
    # Check for OpenAI API key first
    if not check_openai_api_key():
        print("\nOpenAI API key is required to run BLAST. Exiting.")
        sys.exit(1)
    
    # Check if already installed
    already_installed = check_installation_state()
    
    # Install browsers and dependencies if needed
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            if not p.chromium.executable_path.exists():
                install_browsers(quiet=already_installed)
    except Exception:
        install_browsers(quiet=already_installed)
    cli()

if __name__ == '__main__':
    main()