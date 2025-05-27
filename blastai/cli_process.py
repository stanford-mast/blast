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

logger = logging.getLogger('blastai')

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

# Metrics display state
class MetricsDisplay:
    def __init__(self):
        self.initialized = False
        self.lines = 9  # Total lines in metrics display (including empty line)
        self.last_metrics = None

    def reset(self):
        """Reset display state."""
        self.initialized = False
        self.last_metrics = None

# Global metrics display instance
_metrics_display = None

def get_metrics_display():
    """Get or create metrics display instance."""
    global _metrics_display
    if _metrics_display is None:
        _metrics_display = MetricsDisplay()
    return _metrics_display

def format_metrics(metrics=None) -> str:
    """Format metrics as a string."""
    return "\n" + "\n".join([
        "Tasks:",
        f"  Scheduled: {metrics['tasks']['scheduled'] if metrics else 0}",
        f"  Running:   {metrics['tasks']['running'] if metrics else 0}",
        f"  Completed: {metrics['tasks']['completed'] if metrics else 0}",
        "",
        "Resources:",
        f"  Active browsers: {metrics['concurrent_browsers'] if metrics else 0}",
        f"  Memory usage:    {metrics['memory_usage_gb']:.1f} GB" if metrics else "  Memory usage:    0.0 GB",
        f"  Total cost:      ${metrics['total_cost']:.4f}" if metrics else "  Total cost:      $0.0000"
    ])

def update_metrics_display(metrics=None, force_clear=False):
    """Update the metrics display.
    
    Args:
        metrics: Optional metrics data to display
        force_clear: If True, clear display even if not initialized
    """
    global _metrics_display
    _metrics_display = get_metrics_display()
    # Print an empty line first to ensure consistent spacing
    if _metrics_display.initialized or force_clear:
        # Move up only the metrics lines
        print(f"\033[{_metrics_display.lines}A", end='')
        # Clear from current position to end of screen
        print("\033[J", end='')
    print(format_metrics(metrics))
    _metrics_display.initialized = True
    _metrics_display.last_metrics = metrics.copy() if metrics else None

async def display_metrics(client: httpx.AsyncClient, server_port: int):
    """Display and update metrics every second."""
    try:
        # Initialize metrics display but don't print yet
        display = get_metrics_display()
        
        # Wait briefly for server to start
        await asyncio.sleep(0.5)
        
        # Print initial metrics after panel (just one newline)
        print("\n")  # Single line after panel
        print(format_metrics(None))  # Print initial empty metrics
        display.initialized = True  # Mark as initialized for future updates
        
        retries = 0
        max_retries = 3
        retry_delay = 1.0

        # Keep track of shutdown state
        is_shutting_down = False
        
        while True:
            try:
                # Get current metrics
                response = await client.get(
                    f"http://127.0.0.1:{server_port}/metrics",
                    timeout=5.0
                )
                response.raise_for_status()
                metrics = response.json()
                
                # Reset retries on success
                retries = 0
                
                # Always update display on first successful metrics fetch
                if display.last_metrics is None:
                    update_metrics_display(metrics)
                # Then only update when metrics change
                elif metrics != display.last_metrics:
                    update_metrics_display(metrics)
                
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error(f"Failed to fetch metrics after {max_retries} retries: {e}")
                    break
                    
                logger.debug(f"Error fetching metrics (attempt {retries}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)  # Wait before retrying
                
    except asyncio.CancelledError:
        # Just reset display state without clearing screen
        display = get_metrics_display()
        display.reset()

async def run_server_and_frontend(
    server,
    actual_server_port: int,
    actual_web_port: int,
    mode: Optional[str] = None,
    *,
    show_metrics: bool = False
) -> None:
    """Run server and frontend processes.
    
    Args:
        server: Uvicorn server instance
        actual_server_port: Port for BLAST server
        actual_web_port: Port for web UI
        mode: Optional mode ('web', 'cli', 'engine', or None for both)
        show_metrics: Whether to show metrics
    """
    tasks = []
    metrics_client = None
    
    try:
        # Set up metrics if enabled
        if show_metrics and (mode == 'engine' or mode is None):
            metrics_client = httpx.AsyncClient()
            metrics_task = asyncio.create_task(
                display_metrics(metrics_client, actual_server_port)
            )
            tasks.append(metrics_task)

        # Start appropriate services
        if mode is None:
            # Run both server and web UI
            async def run_both():
                frontend_task = asyncio.create_task(
                    run_web_frontend(
                        server_port=actual_server_port,
                        web_port=actual_web_port
                    )
                )
                tasks.append(frontend_task)
                await server.serve()
            tasks.append(asyncio.create_task(run_both()))
        elif mode == 'web':
            tasks.append(asyncio.create_task(run_web_frontend(
                server_port=actual_server_port,
                web_port=actual_web_port
            )))
        elif mode == 'cli':
            tasks.append(asyncio.create_task(run_cli_frontend(actual_server_port)))
        elif mode == 'engine':
            tasks.append(asyncio.create_task(server.serve()))

        # Wait for completion
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error running server: {e}")
            
    except asyncio.CancelledError:
        # Let the parent handle shutdown message
        logger.debug("Received shutdown signal")
        
    finally:
        # Clean up tasks quietly
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait([task], timeout=0.5)
                except Exception as e:
                    logger.debug(f"Error cancelling task: {e}")
                
        # Close client quietly
        if metrics_client:
            try:
                await metrics_client.aclose()
            except Exception as e:
                logger.debug(f"Error closing metrics client: {e}")