"""CLI interface for BLAST."""

# Set environment variables before any imports
import hashlib
import os
import time
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"
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

from .cli_config import setup_environment
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
    
    # Generate instance hash for logging
    instance_hash = hashlib.sha256(f"{time.time()}-{os.getpid()}".encode()).hexdigest()[:8]
    
    # Set up environment and load config
    env_path, config = asyncio.run(setup_environment(env, config))
    
    # Get settings
    settings = Settings.create(**config['settings'])
    
    # Configure logging with instance hash
    if settings.logs_dir:
        logs_dir = Path(settings.logs_dir)
        if not logs_dir.exists():
            logs_dir.mkdir(parents=True)
        setup_logging(settings, instance_hash)
    
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

    # Set up logging flags
    using_logs = bool(settings.logs_dir)
    show_metrics = using_logs
    web_logger = logging.getLogger("web") if using_logs else None

    # Create server with proper logging level and config
    import uvicorn
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

    # Print endpoints with log paths if using logs
    if mode == 'engine':
        if using_logs:
            print(f"Engine: http://127.0.0.1:{actual_server_port} ({logs_dir / f'{instance_hash}.engine.log'})")
        else:
            print(f"Engine: http://127.0.0.1:{actual_server_port}")
    elif mode == 'web':
        if using_logs:
            print(f"Web: http://localhost:{actual_web_port} ({logs_dir / f'{instance_hash}.web.log'})")
        else:
            print(f"Web: http://localhost:{actual_web_port}")
    elif mode is None:
        if using_logs:
            print(f"Engine: http://127.0.0.1:{actual_server_port} ({logs_dir / f'{instance_hash}.engine.log'})")
            print(f"Web: http://localhost:{actual_web_port} ({logs_dir / f'{instance_hash}.web.log'})")
        else:
            print(f"Engine: http://127.0.0.1:{actual_server_port}")
            print(f"Web: http://localhost:{actual_web_port}")

    # Run everything in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
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
            web_logger=web_logger,
            instance_hash=instance_hash
        ))
    except KeyboardInterrupt:
        pass
        # print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        loop.close()

def main():
    """Main entry point for CLI."""
    return cli()  # Return Click's exit code

if __name__ == '__main__':
    main()