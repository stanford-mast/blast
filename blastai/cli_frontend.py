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
            logging.getLogger('blastai').error(f"CLI frontend error: {e}")
            print(f"Error: {e}")  # Keep error visible for user interaction
            continue

async def run_web_frontend(server_port: int, web_port: int):
    """Run web frontend."""
    web_logger = logging.getLogger('web')
    frontend_dir = Path(__file__).parent / 'frontend'
    
    # Check Node.js and npm installation
    if not check_node_installation():
        web_logger.error("Node.js not found")
        return
        
    npm_cmd = check_npm_installation()
    if not npm_cmd:
        web_logger.error("npm not found")
        return
        
    # Install dependencies if needed
    if not (frontend_dir / 'node_modules').exists():
        web_logger.info("Installing frontend dependencies...")
        try:
            result = subprocess.run(
                [npm_cmd, 'install'],
                cwd=frontend_dir,
                capture_output=True,
                text=True
            )
            if result.stdout:
                web_logger.info(result.stdout)
            if result.stderr:
                web_logger.warning(result.stderr)
        except subprocess.CalledProcessError as e:
            web_logger.error(f"Error installing frontend dependencies: {e}")
            return

    # Start frontend process
    try:
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
        web_logger.error(f"Error starting frontend: {e}")
        return

    # Monitor frontend output
    def log_output():
        try:
            for line in process.stdout:
                line = line.strip()
                if line:
                    # Log based on content
                    if any(x in line.lower() for x in ['error:', 'warn:', 'invalid']):
                        web_logger.warning(f"Web: {line}")
                    elif '✓ ready' in line:
                        web_logger.info(f"Web: {line}")
                    else:
                        web_logger.debug(f"Web: {line}")
        except (ValueError, IOError) as e:
            web_logger.error(f"Error reading frontend output: {e}")
    
    output_thread = threading.Thread(target=log_output, daemon=True)
    output_thread.start()

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        # Clean up process first
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            
        # Close stdout to stop output thread
        try:
            process.stdout.close()
        except:
            pass
            
        # Wait briefly for output thread
        output_thread.join(timeout=0.5)
        
        # Now we can raise
        raise
    finally:
        # Final cleanup attempt
        if process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=0.1)
            except:
                pass