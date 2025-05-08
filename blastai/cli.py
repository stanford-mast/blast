"""CLI interface for BLAST."""

import hashlib
import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, Union
from dotenv import load_dotenv

# Set default environment variables
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

# Now import everything else
import rich_click as click
from rich.console import Console
from rich.text import Text
from rich.panel import Panel

# Configure rich-click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True

# Style configuration
click.rich_click.STYLE_COMMANDS = "blue"  # Command names in blue
click.rich_click.STYLE_OPTIONS = "white"  # Option values in white
click.rich_click.STYLE_SWITCH = "white"   # Option switches in white
click.rich_click.STYLE_HEADER = "green"   # Section headers in green
click.rich_click.STYLE_HELPTEXT = "white" # Help text in white
click.rich_click.STYLE_USAGE = "green"    # Usage header in pastel yellow
click.rich_click.STYLE_USAGE_COMMAND = "rgb(255,223,128)" # Command in usage in blue

console = Console()
import httpx
import uvicorn
import asyncio
import threading
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Union
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

@click.group(invoke_without_command=True)
@click.version_option('0.1.6', '-V', '--version', prog_name='BLAST')
@click.pass_context
def cli(ctx):
    """🚀  Browser-LLM Auto-Scaling Technology"""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help(), end="")
        links_panel = Panel(
            "\n".join([
                "🌐  [link=https://blastproject.org]Website[/]",
                "📚  [link=https://docs.blastproject.org]Docs[/]",
                "💬  [link=https://discord.gg/NqrkJwYYh4]Discord[/]",
                "⭐  [link=https://github.com/stanford-mast/blast]github.com/Stanford-MAST/BLAST[/]"
            ]),
            border_style="bright_black",
            title="Support",
            title_align="left",
            # padding=(0, 2)
        )
        console.print(links_panel)
        print()

@cli.command()
@click.argument('command', required=False)
def help(command):
    """Show help for a command."""
    ctx = click.get_current_context()
    if command:
        cmd = cli.get_command(ctx, command)
        if cmd:
            console.print(cmd.get_help(ctx))
        else:
            console.print(f"[red]Error:[/] No such command '[blue]{command}[/]'")
    else:
        console.print(cli.get_help(ctx))

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

@cli.command('serve')
@click.argument('mode', type=click.Choice(['web', 'cli', 'engine']), required=False)
@click.option('--config', type=str, metavar='PATH', help='Path to config file containing constraints and settings')
@click.option('--server-port', type=int, default=8000, metavar='PORT', help='Server port')
@click.option('--web-port', type=int, default=3000, metavar='PORT', help='Web UI port')
@click.option('--env', type=str, metavar='KEY=VALUE,...', help='Environment variables to set (e.g. OPENAI_API_KEY=xxx)')
def serve(config: Optional[str], server_port: int, web_port: int, env: Optional[str], mode: Optional[str] = None):
    """Start BLAST (default: serves engine and web UI)"""
    
    # Load environment variables first
    env_path = load_environment(env)
    
    # Initialize app state with config
    init_app_state(config)
    
    # Get settings that were loaded
    from .server import _settings, _constraints
    if _settings is None or _constraints is None:
        raise RuntimeError("Settings not initialized properly")
    settings = _settings
    constraints = _constraints
    
    # Check for required API keys
    if not check_model_api_key(constraints.llm_model, env_path):
        print("\nRequired API key not found. Exiting.")
        sys.exit(1)
        
    if constraints.llm_model_mini and not check_model_api_key(constraints.llm_model_mini, env_path):
        print("\nRequired API key not found for mini model. Exiting.")
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
    # Print BLAST logo and version
    logo = """              ...........              
          ..........--.......          
       ......--+##########++-...       
     ......-########+------+##-...     
    .....-#######+...........-##-..    
   .....-#######-..............+#+..   
  .....-#######-................+#+..  
 .....-########.................-##-.. 
 .....-#######+..................##+.. 
 .....+########.................-##+.. 
......+########+................###+...
 .....-#########+-............-####-.. 
 ......+###########--......-+#####+... 
 .......##########################.... 
  .......+######################+....  
   .......-+##################+-....   
    .........-+############+--.....    
     ...........----------.......      
       .........................       
          ...................          
              ...........              """
    # console.print(logo)
    # console.print(f"BLAST: Browser-LLM Auto-Scaling Technology v0.1.1")

    # Initialize app state with config (this loads default_config.yaml)
    ctx = click.get_current_context()
    init_app_state(config)
    
    # Get settings that were loaded by init_app_state
    from .server import _settings
    if _settings is None:
        raise RuntimeError("Settings not initialized properly")
    settings = _settings
    
    # Check log level
    if settings.blastai_log_level.upper() not in ['DEBUG']:
        pass  # Warning filters now handled in __init__.py
    
    async def run_web_frontend(*,
        server_port: int,
        web_port: int,
        output_event: Optional[asyncio.Event] = None,
        web_logger: Optional[logging.Logger] = None,
        using_logs: bool = False,
        frontend_ready: Optional[threading.Event] = None
    ):
        """Run just the web frontend.
        
        Args:
            server_port: Port number for the BLAST server
            web_port: Port number for the web UI
            output_event: Event to signal when output occurs (for metrics)
            web_logger: Logger for web UI output when using file logging
            using_logs: Whether to use file logging
            frontend_ready: Event to signal when frontend is ready
        """
        frontend_dir = Path(__file__).parent / 'frontend'
        if frontend_ready is None:
            frontend_ready = threading.Event()
        
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
            # Set environment variables for the frontend process
            frontend_env = os.environ.copy()
            frontend_env['NEXT_PUBLIC_SERVER_PORT'] = str(server_port)
            frontend_env['PORT'] = str(web_port)
            
            process = subprocess.Popen(
                [npm_cmd, 'run', 'dev', f'--port={web_port}'],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=frontend_env
            )
        except subprocess.CalledProcessError as e:
            print(f"Error starting frontend: {e}")
            return

        # Monitor frontend output
        def print_output():
            try:
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        # Signal output if metrics are being shown
                        if output_event:
                            output_event.set()
                            timer = threading.Timer(5.0, output_event.clear)
                            timer.daemon = True
                            timer.start()

                        # Only log important info
                        if any(x in line.lower() for x in ['error:', 'warn:', '✓ ready', 'invalid']):
                            # Log to file if using logs
                            if web_logger:
                                web_logger.info(f"Web: {line}")
                            
                            # Print to terminal if not using logs
                            if not using_logs:
                                print(f"Web: {line}")

                        # Set ready event when frontend is loaded
                        if '✓ ready' in line:
                            frontend_ready.set()

            except (ValueError, IOError) as e:
                error = f"Error reading frontend output: {e}"
                if web_logger:
                    web_logger.error(error)
                if not using_logs:
                    print(error)
        
        output_thread = threading.Thread(target=print_output, daemon=True)
        output_thread.start()

        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
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

    async def display_metrics(client, settings: Settings, server_port: int, output_event: asyncio.Event):
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
            
            # Only move cursor and reprint if no recent output
            if not output_event.is_set():
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
        
        # Wait a bit for server to start
        await asyncio.sleep(2)
        
        while True:
            try:
                # Get metrics from server
                response = await client.get(f"http://127.0.0.1:{server_port}/metrics")
                metrics = response.json()
                print_metrics(metrics)
            except Exception as e:
                print(f"\nError: failed to fetch engine metrics: {e}")
                print_metrics()
                break
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

        # Determine logging and metrics behavior
        using_logs = False
        show_metrics = False
        logs_dir_path = None
        engine_hash = None

        # Get engine instance to get its hash
        from .server import get_engine
        engine = await get_engine()
        engine_hash = engine._instance_hash

        # Determine logging behavior based on settings
        if settings.logs_dir:
            # User specified logs directory in config - always log to files and show metrics
            logs_dir_path = Path(settings.logs_dir)
            if not logs_dir_path.exists():
                logs_dir_path.mkdir(parents=True)
            using_logs = True
            show_metrics = True
        # No need for elif condition since we always want to use logs if logs_dir is set

        # Configure logging if using logs
        log_config = None
        if using_logs:
            # Remove any existing handlers from root logger to prevent double output
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                
            log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "engine_file": {
                    "class": "logging.FileHandler",
                    "filename": str(logs_dir_path / f"{engine_hash}.engine.log"),
                    "formatter": "default",
                },
                "web_file": {
                    "class": "logging.FileHandler",
                    "filename": str(logs_dir_path / f"{engine_hash}.web.log"),
                    "formatter": "default",
                }
            },
            "loggers": {
                "blastai": {
                    "handlers": ["engine_file"],
                    "level": settings.blastai_log_level.upper(),
                    "propagate": False,
                },
                "browser_use": {
                    "handlers": ["engine_file"],
                    "level": settings.browser_use_log_level.upper(),
                    "propagate": False,
                },
                "uvicorn": {
                    "handlers": ["engine_file"],
                    "level": settings.blastai_log_level.upper(),
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["engine_file"],
                    "level": settings.blastai_log_level.upper(),
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["engine_file"],
                    "level": settings.blastai_log_level.upper(),
                    "propagate": False,
                },
                "web": {
                    "handlers": ["web_file"],
                    "level": settings.browser_use_log_level.upper(),
                    "propagate": False,
                }
            }
        }

        # Create server with proper logging level and config
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
            access_log=False,
            log_config=log_config if using_logs else None
        )
        server = uvicorn.Server(server_config)
        server.force_exit = False  # Allow graceful shutdown

        # Print endpoints with log paths if using logs
        if mode == 'engine':
            if using_logs:
                print(f"Engine: http://127.0.0.1:{actual_server_port} ({logs_dir_path / f'{engine_hash}.engine.log'})")
            else:
                print(f"Engine: http://127.0.0.1:{actual_server_port}")
        elif mode == 'web':
            if using_logs:
                print(f"Web: http://localhost:{actual_web_port} ({logs_dir_path / f'{engine_hash}.web.log'})")
            else:
                print(f"Web: http://localhost:{actual_web_port}")
        elif mode is None:
            if using_logs:
                print(f"Engine: http://127.0.0.1:{actual_server_port} ({logs_dir_path / f'{engine_hash}.engine.log'})")
                print(f"Web: http://localhost:{actual_web_port} ({logs_dir_path / f'{engine_hash}.web.log'})")
            else:
                print(f"Engine: http://127.0.0.1:{actual_server_port}")
                print(f"Web: http://localhost:{actual_web_port}")

        # Create metrics task if needed
        metrics_task = None
        metrics_client = None
        output_event = None
        web_logger = None
        try:
            # Set up metrics if enabled
            if show_metrics and (mode == 'engine' or mode is None):
                metrics_client = httpx.AsyncClient()
                output_event = asyncio.Event()
                metrics_task = asyncio.create_task(
                    display_metrics(metrics_client, settings, actual_server_port, output_event)
                )

            # Get web logger if using logs
            if using_logs:
                web_logger = logging.getLogger("web")

            if mode is None:
                # Default behavior: run both server and web UI
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
                            if line:
                                # Signal output if metrics are being shown
                                if output_event:
                                    output_event.set()
                                    timer = threading.Timer(5.0, output_event.clear)
                                    timer.daemon = True
                                    timer.start()

                                # Only log important info
                                if any(x in line.lower() for x in ['error:', 'warn:', '✓ ready', 'invalid']):
                                    # Log to file if using logs
                                    if web_logger:
                                        web_logger.info(f"Web: {line}")
                                    
                                    # Print to terminal if not using logs
                                    if not using_logs:
                                        print(f"Web: {line}")

                                # Set ready event when frontend is loaded
                                if '✓ ready' in line:
                                    frontend_ready.set()

                    except (ValueError, IOError) as e:
                        error = f"Error reading frontend output: {e}"
                        if web_logger:
                            web_logger.error(error)
                        if not using_logs:
                            print(error)
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

            elif mode == 'web':
                await run_web_frontend()
            elif mode == 'cli':
                await run_standalone_cli(actual_server_port)
            elif mode == 'engine':
                # Only run the server
                await server.serve()

        except asyncio.CancelledError:
            if using_logs:
                # Silently shutdown when using logs
                await server.shutdown()
                raise
            else:
                print("\nShutting down server...")
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
    from .models import is_openai_model
    
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

def setup_environment_and_dependencies(env: Optional[str] = None) -> Path:
    """Set up environment variables and install dependencies.
    
    Args:
        env: Optional environment variables from --env parameter
        
    Returns:
        Path to .env file
    """
    # Load environment variables first
    env_path = load_environment(env)
    
    # Initialize app state
    init_app_state(None)
    
    # Get settings and constraints
    from .server import _settings, _constraints
    if _settings is None or _constraints is None:
        raise RuntimeError("Settings or constraints not initialized properly")
    settings = _settings
    constraints = _constraints
    
    # Check for required API keys
    if not check_model_api_key(constraints.llm_model, env_path):
        print("\nRequired API key not found. Exiting.")
        sys.exit(1)
        
    if constraints.llm_model_mini and not check_model_api_key(constraints.llm_model_mini, env_path):
        print("\nRequired API key not found for mini model. Exiting.")
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
        
    return env_path

def main():
    """Main entry point for CLI."""
    return cli()  # Return Click's exit code

if __name__ == '__main__':
    main()