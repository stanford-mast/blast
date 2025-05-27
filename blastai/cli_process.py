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
            tasks.append(asyncio.create_task(
                display_metrics(metrics_client, actual_server_port, output_event)
            ))

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
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.debug("Received CancelledError, cleaning up...")
        # First cancel all tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait briefly for tasks to complete
        try:
            done, pending = await asyncio.wait(tasks, timeout=2.0)
            for task in pending:
                logger.debug(f"Task {task.get_name()} did not complete in time")
        except Exception as e:
            logger.debug(f"Error waiting for tasks: {e}")
            
        # Clean up metrics client if it exists
        if metrics_client:
            try:
                await asyncio.wait_for(metrics_client.aclose(), timeout=1.0)
            except Exception as e:
                logger.debug(f"Error closing metrics client: {e}")
                
        # Don't exit here - let the parent handle it
        return
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return
    finally:
        # Clean up metrics client if we haven't already
        if metrics_client and not metrics_client.is_closed:
            try:
                await asyncio.wait_for(metrics_client.aclose(), timeout=1.0)
            except Exception as e:
                logger.debug(f"Error closing metrics client in finally: {e}")