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
from .logging_setup import should_show_metrics

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
    
    # Set up logging
    setup_logging(settings, engine._instance_hash)
    
    # Find available ports
    actual_server_port = find_available_port(settings.server_port)
    if not actual_server_port:
        print(f"Error: Could not find available port for server (tried ports {settings.server_port}-{settings.server_port+9})")
        return
    
    actual_web_port = find_available_port(settings.web_port)
    if not actual_web_port:
        print(f"Error: Could not find available port for web frontend (tried ports {settings.web_port}-{settings.web_port+9})")
        return
        
    # Configure uvicorn
    import uvicorn
    uvicorn.config.LOGGING_CONFIG = None  # Use our logging
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
    server.install_signal_handlers = lambda: None  # Disable uvicorn's signal handlers

    # Print startup banner
    logs_dir = Path(settings.logs_dir or 'blast-logs')
    # Always show logs since they're essential information
    if mode == 'engine':
        console.print(Panel(
            f"[green]Engine:[/] http://127.0.0.1:{actual_server_port}\n" +
            f"[dim]Logs:[/]   {logs_dir / f'{engine._instance_hash}.engine.log'}",
            title="BLAST Engine",
            border_style="bright_black"
        ))
    elif mode == 'web':
        console.print(Panel(
            f"[green]Web:[/]  http://localhost:{actual_web_port}\n" +
            f"[dim]Logs:[/] {logs_dir / f'{engine._instance_hash}.web.log'}",
            title="BLAST Web UI",
            border_style="bright_black"
        ))
    elif mode is None:
        console.print(Panel(
            f"[green]Engine:[/] http://127.0.0.1:{actual_server_port}  " +
            f"[dim]{logs_dir / f'{engine._instance_hash}.engine.log'}[/]\n" +
            f"[green]Web:[/]    http://localhost:{actual_web_port}  " +
            f"[dim]{logs_dir / f'{engine._instance_hash}.web.log'}[/]",
            border_style="bright_black"
        ))

    # Run everything in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Set up signal handlers
    import signal
    
    def handle_signal(sig, frame):
        """Handle SIGINT/SIGTERM gracefully."""
        if not loop.is_closed():
            print("\nShutting down...\n")  # Add newline to avoid overwriting metrics
            try:
                # Run cleanup synchronously to ensure completion
                loop.run_until_complete(cleanup())
                # Don't close loop here, let finally block handle it
            except:
                sys.exit(0)
            
    async def cleanup():
        """Clean up resources gracefully."""
        try:
            # Cancel all tasks except cleanup
            tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            
            if tasks:
                # Wait longer for tasks to cancel
                await asyncio.wait(tasks, timeout=5.0)
                
            # Give subprocess cleanup a chance
            await asyncio.sleep(0.5)
            
            # Run event loop one more time to allow subprocess cleanup
            loop.stop()
            loop.run_forever()
            
            # Now we can close transports
            for task in tasks:
                if hasattr(task, 'transport'):
                    try:
                        task.transport.close()
                    except:
                        pass
                        
            # Finally stop the loop
            loop.stop()
        except Exception as e:
            logging.getLogger('blastai').error(f"Error during cleanup: {e}")
            
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
                actual_server_port=actual_server_port,
                actual_web_port=actual_web_port,
                mode=mode,
                show_metrics=True  # Always show metrics
            ))
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nShutting down...\n")  # Add newline to avoid overwriting metrics
            loop.run_until_complete(cleanup())
        except Exception as e:
            # Handle any errors
            print("\nShutting down...\n")  # Add newline to avoid overwriting metrics
            if e.__class__.__name__ in ('SystemExit', 'KeyboardInterrupt'):
                loop.run_until_complete(cleanup())
            else:
                loop.run_until_complete(cleanup())
    finally:
        try:
            # Run loop one final time to clean up any remaining transports
            if not loop.is_closed():
                loop.run_until_complete(asyncio.sleep(0))
                loop.close()
        except Exception as e:
            logging.getLogger('blastai').error(f"Error closing loop: {e}")

def main():
    """Main entry point for CLI."""
    return cli()  # Return Click's exit code

if __name__ == '__main__':
    main()