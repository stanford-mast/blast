"""Frontend management for BLAST CLI."""

import os
import sys
import asyncio
import logging
import threading
import subprocess
from pathlib import Path
from typing import Optional
from openai import OpenAI

from .cli_installation import check_node_installation, check_npm_installation

logger = logging.getLogger(__name__)

async def run_cli_frontend(server_port: int):
    """Run CLI frontend for interacting with BLAST.
    
    Args:
        server_port: Port number for the BLAST server
    """
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

async def run_web_frontend(*,
    server_port: int,
    web_port: int,
    output_event: Optional[asyncio.Event] = None,
    web_logger: Optional[logging.Logger] = None,
    using_logs: bool = False,
    frontend_ready: Optional[threading.Event] = None
):
    """Run web frontend.
    
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
        # Log installation start
        msg = "Installing frontend dependencies..."
        if web_logger:
            web_logger.info(msg)
        if not using_logs:
            print(msg)
            
        try:
            # Capture npm install output
            result = subprocess.run(
                [npm_cmd, 'install'], 
                cwd=frontend_dir, 
                capture_output=True, 
                text=True
            )
            
            # Log npm output
            if result.stdout:
                if web_logger:
                    web_logger.info(result.stdout)
                if not using_logs:
                    print(result.stdout)
                    
            if result.stderr:
                if web_logger:
                    web_logger.warning(result.stderr)
                if not using_logs:
                    print(result.stderr, file=sys.stderr)
                    
        except subprocess.CalledProcessError as e:
            error = f"Error installing frontend dependencies: {e}"
            if web_logger:
                web_logger.error(error)
            if not using_logs:
                print(error)
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
        error = f"Error starting frontend: {e}"
        if web_logger:
            web_logger.error(error)
        if not using_logs:
            print(error)
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

                    # Always log important info
                    if any(x in line.lower() for x in ['error:', 'warn:', 'invalid']):
                        if web_logger:
                            web_logger.warning(f"Web: {line}")
                        print(f"Web: {line}")
                    elif '✓ ready' in line:
                        # Always show ready message
                        if web_logger:
                            web_logger.info(f"Web: {line}")
                        print(f"Web: {line}")
                        frontend_ready.set()
                    else:
                        # Log other output based on settings
                        if web_logger:
                            web_logger.debug(f"Web: {line}")
                        elif not using_logs:
                            print(f"Web: {line}")

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