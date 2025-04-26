# Configure environment variables and logging before any imports
import os
import sys
import logging

# Set environment variables before any browser-use imports
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["BROWSER_USE_LOGGING_LEVEL"] = "error"

# Configure root logger to error by default
logging.basicConfig(
    level=logging.ERROR,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Configure all loggers to error level
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('uvicorn').setLevel(logging.ERROR)
logging.getLogger('uvicorn.access').setLevel(logging.ERROR)

# Import server first to ensure logging is configured before browser-use import
from .server import app, init_app_state

"""CLI interface for BLAST."""

# Import other modules after server to ensure proper logging setup
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

def find_executable(*names: str) -> Optional[str]:
    """Find the first available executable from the given names.
    
    Args:
        *names: Executable names to search for (e.g., 'node', 'node.exe')
        
    Returns:
        Path to executable if found, None otherwise
    """
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None

def check_node_installation() -> Optional[str]:
    """Check if Node.js is installed and available.
    
    Returns:
        Path to node executable if found, None otherwise
    """
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
    """Check if npm is installed and available.
    
    Returns:
        Path to npm executable if found, None otherwise
    """
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

async def run_cli_frontend():
    """Run CLI frontend for interacting with BLAST."""
    client = OpenAI(
        api_key="not-needed",
        base_url="http://127.0.0.1:8000"
    )
    
    print("> ", end='', flush=True)
    
    previous_response_id = None
    
    while True:
        try:
            task = input("> ")
            if task.lower() == 'exit':
                break
                
            stream = client.responses.create(
                model="gpt-4.1-mini",
                input=task,
                stream=True,
                previous_response_id=previous_response_id
            )
            
            for event in stream:
                if event.type == "response.completed":
                    previous_response_id = event.response.id
                elif event.type == "response.output_text.delta":
                    if ' ' in event.delta:  # Skip screenshots
                        print(event.delta, end='', flush=True)
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

async def run_cli_server(server):
    """Run server with CLI frontend."""
    server.should_exit = lambda: False
    server.startup_complete = lambda: (
        asyncio.get_event_loop().create_task(run_cli_frontend())
    )
    await server.serve()

@click.group()
def cli():
    """BLAST CLI tool for browser automation."""
    pass

@cli.command()
@click.option('--config', type=str, help='Path to config YAML file')
@click.option('--no-metrics-output', is_flag=True, help='Disable metrics output')
@click.argument('component', type=click.Choice(['web', 'cli', 'engine']), required=False)
def serve(config: Optional[str], no_metrics_output: bool, component: Optional[str]):
    """Start BLAST components. If no component specified, serves both backend and web frontend."""
    # For backward compatibility, if no component specified, default to web frontend
    frontend = 'web' if component is None else component
    
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
        output_thread = None
        def print_output():
            try:
                for line in process.stdout:
                    pass  # Just consume output
            except (ValueError, IOError):
                pass
        
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
                # Wait for process to terminate with timeout
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                process.kill()
                process.wait()

    async def run_standalone_cli():
        """Run just the CLI frontend."""
        client = OpenAI(
            api_key="not-needed",
            base_url="http://127.0.0.1:8000"
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
                        model="gpt-4.1-mini",
                        input=task,
                        stream=True,
                        previous_response_id=previous_response_id
                    )
                    
                    for event in stream:
                        if event.type == "response.completed":
                            previous_response_id = event.response.id
                        elif event.type == "response.output_text.delta":
                            if ' ' in event.delta:  # Skip screenshots
                                print(event.delta, flush=True)
                    print()
                    
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

    async def display_metrics(client):
        """Display and update metrics every 5s."""
        # Print initial metrics
        print("Tasks scheduled/running/completed: 0/0/0")
        print("Concurrent browsers: 0")
        print("Total memory usage: 0.0 GB")
        print("Total cost: $0.00")

        while True:
            try:
                response = await client.get("http://127.0.0.1:8000/metrics")
                metrics = response.json()
                
                # Clear previous lines and update metrics
                print("\033[2K\033[1G\033[4A", end='')  # Clear line, move to start, up 4 lines
                print(f"Tasks scheduled/running/completed: {metrics['tasks']['scheduled']}/{metrics['tasks']['running']}/{metrics['tasks']['completed']}")
                print(f"Concurrent browsers: {metrics['concurrent_browsers']}")
                print(f"Total memory usage: {metrics['memory_usage_gb']:.1f} GB")
                print(f"Total cost: ${metrics['total_cost']:.2f}", flush=True)
                
            except Exception:
                # On error, just update timestamp
                print("\033[2K\033[1G\033[4A", end='')  # Clear line, move to start, up 4 lines
                print("Tasks scheduled/running/completed: 0/0/0")
                print("Concurrent browsers: 0")
                print("Total memory usage: 0.0 GB")
                print("Total cost: $0.00", flush=True)
                
            await asyncio.sleep(5)

    async def run_server_and_frontend():
        """Run server and frontend concurrently."""
        # Create server
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="error",
            reload=False,
            workers=1,
            lifespan="on",
            timeout_keep_alive=5,
            timeout_graceful_shutdown=10,
            access_log=False
        )
        server = uvicorn.Server(config)
        server.force_exit = False  # Allow graceful shutdown

        # Print server endpoint and start metrics if enabled
        if component == 'engine':
            print("Server: http://127.0.0.1:8000")
        elif component == 'web':
            print("Web: http://localhost:3000")
        elif component is None:
            print("Server: http://127.0.0.1:8000")
            print("Web: http://localhost:3000")
        if not no_metrics_output and component != 'web':
            print("\n\n\n\n")  # Four lines for metrics
            async with httpx.AsyncClient() as client:
                metrics_task = asyncio.create_task(display_metrics(client))

        if component is None:
            # Default behavior: run both backend and web frontend
            frontend_dir = Path(__file__).parent / 'frontend'
            
            # Check Node.js and npm installation
            if not check_node_installation():
                await run_cli_server(server)
                return
                
            npm_cmd = check_npm_installation()
            if not npm_cmd:
                await run_cli_server(server)
                return
                
            # Install dependencies if needed
            if not (frontend_dir / 'node_modules').exists():
                try:
                    subprocess.run([npm_cmd, 'install'], cwd=frontend_dir, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error installing frontend dependencies: {e}")
                    await run_cli_server(server)
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
                await run_cli_server(server)
                return

            # Monitor frontend output
            def print_output():
                for line in process.stdout:
                    pass  # Just consume output
            threading.Thread(target=print_output, daemon=True).start()

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
            await run_standalone_cli()
        elif component == 'engine':
            # Only run the backend server
            try:
                await server.serve()
            except asyncio.CancelledError:
                # Let server handle its own shutdown
                await server.shutdown()
                raise
    
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

def main():
    """Main entry point for CLI."""
    cli()

if __name__ == '__main__':
    main()