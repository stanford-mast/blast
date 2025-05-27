"""Process management for BLAST CLI."""

import os
import sys
import asyncio
import logging
import threading
import subprocess
from pathlib import Path
from typing import Optional
import httpx

from .cli_frontend import run_cli_frontend, run_web_frontend

logger = logging.getLogger(__name__)

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

async def display_metrics(client: httpx.AsyncClient, server_port: int, output_event: asyncio.Event):
    """Display and update metrics every 5s.
    
    Args:
        client: HTTP client for making requests
        server_port: Port number for the BLAST server
        output_event: Event to signal when output should be suppressed
    """
    # Get logger
    engine_logger = logging.getLogger('blastai')
    
    # Constants for metrics display
    METRICS_LINES = 10  # Including blank line at start
    BLANK_METRICS = {
        'tasks': {'scheduled': 0, 'running': 0, 'completed': 0},
        'concurrent_browsers': 0,
        'memory_usage_gb': 0.0,
        'total_cost': 0.00
    }
    
    def print_metrics(metrics=None, error_occurred=False, force_clear=False):
        """Print metrics in a consistent format."""
        if metrics is None:
            metrics = BLANK_METRICS
            
        # Handle different display states
        if force_clear:
            # Clear entire metrics section
            print("\033[J", end='')
            print_metrics.initialized = False
            return
            
        if error_occurred or output_event.is_set():
            if print_metrics.initialized:
                # Clear metrics but leave space
                print("\033[J", end='')
                print("\n\nTasks:")
                print("  Scheduled: 0")
                print("  Running:   0")
                print("  Completed: 0")
                print("\nResources:")
                print("  Active browsers: 0")
                print("  Memory usage:    0.0 GB")
                print("  Total cost:      $0.0000", flush=True)
            return
            
        # Normal metrics update
        if print_metrics.initialized:
            print(f"\033[{METRICS_LINES}A\033[J", end='')
        print_metrics.initialized = True
        
        # Print metrics
        print("\nTasks:")
        print(f"  Scheduled: {metrics['tasks']['scheduled']}")
        print(f"  Running:   {metrics['tasks']['running']}")
        print(f"  Completed: {metrics['tasks']['completed']}")
        print("\nResources:")
        print(f"  Active browsers: {metrics['concurrent_browsers']}")
        print(f"  Memory usage:    {metrics['memory_usage_gb']:.1f} GB")
        print(f"  Total cost:      ${metrics['total_cost']:.4f}", flush=True)
    
    # Initialize the print_metrics function state
    print_metrics.initialized = False
    
    try:
        # Wait for server to start, checking every 0.5s
        retries = 10
        while retries > 0:
            try:
                response = await client.get(f"http://127.0.0.1:{server_port}/metrics")
                metrics = response.json()
                print_metrics(metrics)
                break
            except Exception:
                retries -= 1
                if retries == 0:
                    engine_logger.error("Failed to connect to server for metrics")
                    return
                await asyncio.sleep(0.5)
        
        # Server is up, start regular updates
        while not output_event.is_set():
            try:
                response = await client.get(f"http://127.0.0.1:{server_port}/metrics")
                metrics = response.json()
                print_metrics(metrics)
                await asyncio.sleep(5)
            except Exception as e:
                if not output_event.is_set():  # Only log if not shutting down
                    engine_logger.error(f"Failed to fetch metrics: {e}")
                break
    except asyncio.CancelledError:
        # Clean shutdown, don't log
        pass
    except Exception as e:
        if not output_event.is_set():  # Only log if not shutting down
            engine_logger.error(f"Error in metrics display: {e}")
    finally:
        # Clear metrics display on exit
        if print_metrics.initialized:
            print_metrics(force_clear=True)

async def run_server_and_frontend(
    server,
    settings,
    actual_server_port: int,
    actual_web_port: int,
    mode: Optional[str] = None,
    *,
    using_logs: bool = False,
    show_metrics: bool = False,
    web_logger: Optional[logging.Logger] = None
) -> None:
    """Run server and frontend processes.
    
    Args:
        server: Uvicorn server instance
        settings: Settings instance
        actual_server_port: Port for BLAST server
        actual_web_port: Port for web UI
        mode: Optional mode ('web', 'cli', 'engine', or None for both)
        using_logs: Whether to use file logging
        show_metrics: Whether to show metrics
        web_logger: Optional logger for web UI output
    """
    # Create tasks list to track all tasks
    tasks = []
    metrics_client = None
    
    try:
        # Set up metrics if enabled
        output_event = None
        
        if show_metrics and (mode == 'engine' or mode is None):
            metrics_client = httpx.AsyncClient()
            output_event = asyncio.Event()
            
            # Print initial metrics before starting display task
            print("\nTasks:")
            print("  Scheduled: 0")
            print("  Running:   0")
            print("  Completed: 0")
            print("\nResources:")
            print("  Active browsers: 0")
            print("  Memory usage:    0.0 GB")
            print("  Total cost:      $0.0000", flush=True)
            
            # Create metrics display task
            metrics_task = asyncio.create_task(
                display_metrics(metrics_client, actual_server_port, output_event)
            )
            metrics_task.set_name('metrics_display')
            tasks.append(metrics_task)

        if mode is None:
            # Default behavior: run both server and web UI
            frontend_ready = threading.Event()
            
            async def run_both():
                frontend_task = asyncio.create_task(
                    run_web_frontend(
                        server_port=actual_server_port,
                        web_port=actual_web_port,
                        output_event=output_event,
                        web_logger=web_logger,
                        using_logs=using_logs,
                        frontend_ready=frontend_ready
                    )
                )
                tasks.append(frontend_task)
                await server.serve()
                
            tasks.append(asyncio.create_task(run_both()))
            
        elif mode == 'web':
            tasks.append(asyncio.create_task(run_web_frontend(
                server_port=actual_server_port,
                web_port=actual_web_port,
                output_event=output_event,
                web_logger=web_logger,
                using_logs=using_logs
            )))
        elif mode == 'cli':
            tasks.append(asyncio.create_task(run_cli_frontend(actual_server_port)))
        elif mode == 'engine':
            tasks.append(asyncio.create_task(server.serve()))

        # Wait for all tasks to complete
        # Get engine logger
        engine_logger = logging.getLogger('blastai')
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            # Log error but continue running
            engine_logger.error(f"Error in task: {e}", exc_info=True)
            # Update metrics display
            if output_event:
                output_event.set()
                # Show zeros but keep display space
                print_metrics(error_occurred=True)
            
    except asyncio.CancelledError:
        engine_logger.debug("Received CancelledError, cleaning up...")
        if output_event:
            output_event.set()
            # Clear metrics display completely
            print_metrics(force_clear=True)
        
    except Exception as e:
        # Log error but continue running
        engine_logger.error(f"Server error: {e}", exc_info=True)
        if output_event:
            output_event.set()
            # Show zeros but keep display space
            print_metrics(error_occurred=True)
        
    finally:
        # Always clean up resources
        engine_logger = logging.getLogger('blastai')
        
        # Set output event to stop metrics display
        if output_event:
            output_event.set()
            
        try:
            # Find and kill any running npm processes
            import psutil
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    if 'npm' in child.name().lower() or 'node' in child.name().lower():
                        child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            # Wait for processes to terminate
            psutil.wait_procs(children, timeout=1)
        except Exception as e:
            engine_logger.debug(f"Error cleaning up processes: {e}")
            
        # Cancel all tasks first
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    task._coro.close()
                except:
                    pass
        
        # Wait briefly for tasks to cancel
        if tasks:
            try:
                await asyncio.wait(tasks, timeout=0.5)
            except:
                pass
        
        # Clean up metrics client
        if metrics_client:
            try:
                await asyncio.wait_for(metrics_client.aclose(), timeout=0.5)
            except:
                try:
                    metrics_client._client.close()
                except:
                    pass