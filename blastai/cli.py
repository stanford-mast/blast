"""CLI interface for BLAST."""

# Set environment variables before any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore::ResourceWarning"
os.environ["BROWSER_USE_LOGGING_LEVEL"] = "error"  # Default to error level

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Set up logging before other imports
from .config import Settings
from .logging_setup import setup_logging
setup_logging(Settings())

import rich_click as click
from rich.console import Console
from rich.panel import Panel

from .cli_config import setup_serving_environment
from .cli_process import (
    find_available_port,
    run_server_and_frontend
)

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

@cli.command('serve')
@click.argument('mode', type=click.Choice(['web', 'cli', 'engine']), required=False)
@click.option('--config', type=str, metavar='PATH', help='Path to config file containing constraints and settings')
@click.option('--env', type=str, metavar='KEY=VALUE,...', help='Environment variables to set (e.g. OPENAI_API_KEY=xxx)')
def serve(config: Optional[str], env: Optional[str], mode: Optional[str] = None):
    """Start BLAST (default: serves engine and web UI)"""
    
    # Set up environment and create engine for config
    # NOTE: This engine should only be used for getting configuration, it shouldn't actually be
    # started (that's the responsibility of the server process)
    env_path, engine = asyncio.run(setup_serving_environment(env, config))
    
    # Get settings
    settings = engine.settings
    
    # Find available ports
    actual_server_port = find_available_port(settings.server_port)
    if not actual_server_port:
        print(f"\nError: Could not find available port for server (tried ports {settings.server_port}-{settings.server_port+9})")
        return
    
    actual_web_port = find_available_port(settings.web_port)
    if not actual_web_port:
        print(f"\nError: Could not find available port for web frontend (tried ports {settings.web_port}-{settings.web_port+9})")
        return

    # Determine logging behavior
    using_logs = False
    show_metrics = False
    web_logger = None

    if settings.logs_dir:
        # User specified logs directory in config - always log to files and show metrics
        logs_dir = Path(settings.logs_dir)
        if not logs_dir.exists():
            logs_dir.mkdir(parents=True)
        using_logs = True
        show_metrics = True

        # Configure logging with engine hash
        setup_logging(settings, engine._instance_hash)
        web_logger = logging.getLogger("web")

    # Create server with proper logging level and config
    import uvicorn
    # Configure uvicorn to use our logging
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
        log_config=None  # Use our logging config instead of uvicorn's
    )
    # Create server with custom error handler
    server = uvicorn.Server(server_config)
    server.force_exit = False  # Allow graceful shutdown
    
    # Override error logger to use our logger
    error_logger = logging.getLogger('blastai')
    server.install_signal_handlers = lambda: None  # Disable uvicorn's signal handlers

    # Print startup banner
    if mode == 'engine':
        console.print(Panel(
            f"[green]Engine:[/] http://127.0.0.1:{actual_server_port}\n" +
            (f"[dim]Logs:[/]   {logs_dir / f'{engine._instance_hash}.engine.log'}" if using_logs else ""),
            title="BLAST Engine",
            border_style="bright_black"
        ))
    elif mode == 'web':
        console.print(Panel(
            f"[green]Web:[/]  http://localhost:{actual_web_port}\n" +
            (f"[dim]Logs:[/] {logs_dir / f'{engine._instance_hash}.web.log'}" if using_logs else ""),
            title="BLAST Web UI",
            border_style="bright_black"
        ))
    elif mode is None:
        console.print(Panel(
            f"[green]Engine:[/] http://127.0.0.1:{actual_server_port}" +
            (f"  [dim]{logs_dir / f'{engine._instance_hash}.engine.log'}[/]\n" if using_logs else "\n") +
            f"[green]Web:[/]    http://localhost:{actual_web_port}" +
            (f"  [dim]{logs_dir / f'{engine._instance_hash}.web.log'}[/]" if using_logs else ""),
            # title="BLAST",
            border_style="bright_black"
        ))

    # Run everything in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Configure logging handler for uvicorn errors
    error_logger = logging.getLogger('blastai')
    
    # Set up signal handlers
    import signal
    
    def handle_signal(sig, frame):
        """Handle SIGINT/SIGTERM gracefully."""
        if not loop.is_closed():
            error_logger.info("Received shutdown signal, stopping...")
            
            # Clear screen
            print("\033[J", end='')
            
            if using_logs:
                console.print("[yellow]Shutting down...[/]")
            
            # Force cleanup
            try:
                loop.create_task(cleanup(force=True))
            except:
                # If event loop is broken, force exit
                sys.exit(0)
            
    async def cleanup(force: bool = False):
        """Clean up resources gracefully.
        
        Args:
            force: If True, force close resources without waiting
        """
        try:
            # Cancel all tasks except current
            tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
            
            # Cancel all tasks first
            for task in tasks:
                task.cancel()
            
            if tasks:
                # Wait briefly for tasks to cancel
                try:
                    done, pending = await asyncio.wait(tasks, timeout=1.0 if not force else 0.1)
                    
                    # Force close any pending tasks
                    for task in pending:
                        try:
                            task._coro.close()
                        except:
                            pass
                            
                    # Check for task errors
                    for task in done:
                        try:
                            exc = task.exception()
                            if exc and not isinstance(exc, asyncio.CancelledError):
                                name = task.get_name() or 'unknown'
                                error_logger.debug(f"Task {name} failed: {exc}")
                        except:
                            pass
                except:
                    if not force:
                        error_logger.debug("Error waiting for tasks to cancel")
            
            # Cleanup asyncgens with timeout
            try:
                await asyncio.wait_for(loop.shutdown_asyncgens(),
                                     timeout=1.0 if not force else 0.1)
            except:
                if not force:
                    error_logger.debug("Error shutting down asyncgens")
            
            # Stop loop
            try:
                loop.stop()
            except:
                pass
                
        except Exception as e:
            # Only log if not forced shutdown
            if not force and loop.is_running():
                error_logger.error(f"Error during cleanup: {e}")
            
    # Install signal handlers for all relevant signals
    signals = (
        signal.SIGINT,   # Ctrl+C
        signal.SIGTERM,  # Termination request
        signal.SIGHUP,   # Terminal closed
        signal.SIGQUIT,  # Quit program
        signal.SIGABRT,  # Abort
    )
    
    for sig in signals:
        try:
            signal.signal(sig, handle_signal)
        except (AttributeError, ValueError):
            # Some signals might not be available on all platforms
            pass
    
    try:
        try:
            # Run server and frontend
            loop.run_until_complete(run_server_and_frontend(
                server=server,
                settings=settings,
                actual_server_port=actual_server_port,
                actual_web_port=actual_web_port,
                mode=mode,
                using_logs=using_logs,
                show_metrics=show_metrics,
                web_logger=web_logger
            ))
        except KeyboardInterrupt:
            # Handle Ctrl+C
            error_logger.info("Shutting down...")
            loop.run_until_complete(cleanup())
            print("\nServer stopped cleanly.")
        except Exception as e:
            # Log error but don't crash
            error_logger.error("Server error", exc_info=True)
            
            # Only show error panel if server is stopping
            if e.__class__.__name__ in ('SystemExit', 'KeyboardInterrupt'):
                # Clear screen and show error
                print("\033[J", end='')
                if using_logs:
                    console.print(Panel(
                        f"[red]Server stopped.[/]\nPlease check [blue]{logs_dir / f'{engine._instance_hash}.engine.log'}[/] for details.",
                        title="Server Stopped",
                        border_style="red"
                    ))
                # Force cleanup on shutdown
                loop.run_until_complete(cleanup(force=True))
                sys.exit(0)
            else:
                # For task errors, just log and continue
                loop.run_until_complete(cleanup(force=False))
    finally:
        try:
            if not loop.is_closed():
                loop.close()
        except Exception as e:
            error_logger.error(f"Error closing loop: {e}")

def main():
    """Main entry point for CLI."""
    return cli()  # Return Click's exit code

if __name__ == '__main__':
    main()