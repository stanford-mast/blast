"""CLI interface for BLAST."""

# Set environment variables before any imports
import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore::ResourceWarning,ignore::DeprecationWarning"
os.environ["BROWSER_USE_LOGGING_LEVEL"] = "error"  # Default to error level

# Add environment variable to disable browser-use's own logging setup
os.environ["BROWSER_USE_DISABLE_LOGGING"] = "true"

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import rich_click as click
from rich.console import Console
from rich.panel import Panel

from .cli_config import setup_serving_environment
from .cli_process import find_available_port, run_server_and_frontend
from .logging_setup import setup_logging

# Configure rich-click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True

# Style configuration
click.rich_click.STYLE_COMMANDS = "blue"  # Command names in blue
click.rich_click.STYLE_OPTIONS = "white"  # Option values in white
click.rich_click.STYLE_SWITCH = "white"  # Option switches in white
click.rich_click.STYLE_HEADER = "green"  # Section headers in green
click.rich_click.STYLE_HELPTEXT = "white"  # Help text in white
click.rich_click.STYLE_USAGE = "green"  # Usage header in pastel yellow
click.rich_click.STYLE_USAGE_COMMAND = "rgb(255,223,128)"  # Command in usage in blue

console = Console()


@click.group(invoke_without_command=True)
@click.version_option("0.1.6", "-V", "--version", prog_name="BLAST")
@click.pass_context
def cli(ctx):
    """🚀  Browser-LLM Auto-Scaling Technology"""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help(), end="")
        links_panel = Panel(
            "\n".join(
                [
                    "🌐  [link=https://blastproject.org]Website[/]",
                    "📚  [link=https://docs.blastproject.org]Docs[/]",
                    "💬  [link=https://discord.gg/NqrkJwYYh4]Discord[/]",
                    "⭐  [link=https://github.com/stanford-mast/blast]github.com/Stanford-MAST/BLAST[/]",
                ]
            ),
            border_style="bright_black",
            title="Support",
            title_align="left",
        )
        console.print(links_panel)
        print()


@cli.command()
@click.argument("command", required=False)
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


@cli.command("serve")
@click.argument("mode", type=click.Choice(["web", "cli", "engine"]), required=False)
@click.option("--config", type=str, metavar="PATH", help="Path to config file containing constraints and settings")
@click.option("--env", type=str, metavar="KEY=VALUE,...", help="Environment variables to set (e.g. OPENAI_API_KEY=xxx)")
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
        print(
            f"Error: Could not find available port for server (tried ports {settings.server_port}-{settings.server_port + 9})"
        )
        return

    actual_web_port = find_available_port(settings.web_port)
    if not actual_web_port:
        print(
            f"Error: Could not find available port for web frontend (tried ports {settings.web_port}-{settings.web_port + 9})"
        )
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
        log_config=None,  # Use our logging config instead of uvicorn's
    )

    # Create server with custom error handler
    server = uvicorn.Server(server_config)
    server.force_exit = False  # Allow graceful shutdown
    server.install_signal_handlers = lambda: None  # Disable uvicorn's signal handlers

    # Print startup banner
    logs_dir = Path(settings.logs_dir or "blast-logs")
    # Always show logs since they're essential information
    if mode == "engine":
        console.print(
            Panel(
                f"[green]Engine:[/] http://127.0.0.1:{actual_server_port}\n"
                + f"[dim]Logs:[/]   {logs_dir / f'{engine._instance_hash}.engine.log'}",
                title="BLAST Engine",
                border_style="bright_black",
            )
        )
    elif mode == "web":
        console.print(
            Panel(
                f"[green]Web:[/]  http://localhost:{actual_web_port}\n"
                + f"[dim]Logs:[/] {logs_dir / f'{engine._instance_hash}.web.log'}",
                title="BLAST Web UI",
                border_style="bright_black",
            )
        )
    elif mode is None:
        console.print(
            Panel(
                f"[green]Engine:[/] http://127.0.0.1:{actual_server_port}  "
                + f"[dim]{logs_dir / f'{engine._instance_hash}.engine.log'}[/]\n"
                + f"[green]Web:[/]    http://localhost:{actual_web_port}  "
                + f"[dim]{logs_dir / f'{engine._instance_hash}.web.log'}[/]",
                border_style="bright_black",
            )
        )

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
                if hasattr(task, "transport"):
                    try:
                        task.transport.close()
                    except:
                        pass

            # Finally stop the loop
            loop.stop()
        except Exception as e:
            logging.getLogger("blastai").error(f"Error during cleanup: {e}")

    # Install signal handlers for all relevant signals
    signals = (
        signal.SIGINT,  # Ctrl+C
        signal.SIGTERM,  # Termination request
        signal.SIGHUP,  # Terminal closed
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
            loop.run_until_complete(
                run_server_and_frontend(
                    server=server,
                    actual_server_port=actual_server_port,
                    actual_web_port=actual_web_port,
                    mode=mode,
                    show_metrics=True,  # Always show metrics
                )
            )
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nShutting down...\n")  # Add newline to avoid overwriting metrics
            loop.run_until_complete(cleanup())
        except Exception as e:
            # Handle any errors
            print("\nShutting down...\n")  # Add newline to avoid overwriting metrics
            if e.__class__.__name__ in ("SystemExit", "KeyboardInterrupt"):
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
            logging.getLogger("blastai").error(f"Error closing loop: {e}")


@cli.command('crawl')
@click.option('--url', required=True, type=str, help='URL to crawl for tool generation')
@click.option('--smcp-registry', required=True, type=str, help='Path to SMCP registry JSON file (will be created or updated)')
@click.option('--prompt', required=True, type=str, help='Task description for tool synthesis (e.g., "Create tools to list and filter items")')
@click.option('--user-id', type=str, help='User identifier for persistent browser profile (cookies/sessions)')
def crawl(url: str, smcp_registry: str, prompt: str, user_id: str):
    """Generate SMCP tools by crawling a website"""
    
    # Enable standalone logging mode
    from .logging_setup import enable_standalone_mode
    enable_standalone_mode(browser_use_log_level="INFO")
    
    from .agents import Agent, AgentExecutor
    from pathlib import Path
    import os
    
    async def run_crawl():
        registry_path = Path(smcp_registry)
        
        # Load existing SMCP tools from registry if it exists
        if registry_path.exists():
            console.print(f"[blue]Loading existing tools from:[/] {smcp_registry}")
            # from_smcp_registry loads Agent with only SMCP tools from the registry
            base_agent = Agent.from_smcp_registry(str(registry_path))
            console.print(f"[blue]Loaded {len(base_agent.tools)} existing SMCP tools[/]")
        else:
            console.print(f"[blue]No existing registry - starting from scratch[/]")
            base_agent = Agent(description="", tools=[])
        
        # Add ask_human_cli to base agent before deriving synthesis agent
        from .agents.models import CoreTool
        base_agent.add_tool(CoreTool(name="ask_human_cli"))
        
        # Derive synthesis agent (adds update_smcp_tool, remove_smcp_tool, list_smcp_tools, ask_html)
        # and includes all tools from base_agent (SMCP tools + ask_human_cli)
        synthesis_agent = base_agent.derive_synthesis_agent()
        console.print(f"[blue]Created synthesis agent with {len(synthesis_agent.tools)} tools (SMCP + core tools)[/]")
        
        # Create executor with optional user_id for persistent browser profile
        if user_id:
            console.print(f"[blue]Using persistent browser profile for user:[/] {user_id}")
        executor = AgentExecutor(synthesis_agent, user_id=user_id)
        
        try:
            console.print(f"\n[green]Running synthesis agent...[/]")
            console.print(f"[dim]URL:[/] {url}")
            console.print(f"[dim]Task:[/] {prompt}\n")
            
            # Run synthesis agent
            result = await executor.run(prompt, mode="loop", initial_url=url)
            
            console.print(f"\n[green]✓ Synthesis complete![/]")
            console.print(f"[dim]Result:[/] {result}")
            
            # Get SMCP tools created (exclude core tools)
            from .agents.models import ToolExecutorType
            smcp_tools = [
                tool for tool in synthesis_agent.tools
                if hasattr(tool, 'tool_executor_type') and tool.tool_executor_type == ToolExecutorType.SMCP
            ]
            
            console.print(f"\n[green]Generated {len(smcp_tools)} SMCP tools:[/]")
            for tool in smcp_tools:
                console.print(f"  [blue]•[/] {tool.name}: {tool.description}")
            
            # Save only SMCP tools to registry using to_smcp_registry
            # This filters out CoreTools and saves just the SMCP tools
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            synthesis_agent.to_smcp_registry(str(registry_path))
            
            console.print(f"\n[green]✓ Saved {len(smcp_tools)} SMCP tools to:[/] {smcp_registry}\n")
            
        except Exception as e:
            console.print(f"\n[red]✗ Error during tool generation:[/] {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            await executor.cleanup()
    
    # Run async task
    asyncio.run(run_crawl())


@cli.command('run')
@click.option('--url', required=True, type=str, help='URL to navigate to')
@click.option('--smcp-registry', type=str, help='Path to SMCP registry JSON file (optional)')
@click.option('--prompt', required=True, type=str, help='Task to execute (e.g., "List all leave requests")')
@click.option('--user-id', type=str, help='User identifier for persistent browser profile (cookies/sessions)')
def run(url: str, smcp_registry: str, prompt: str, user_id: str):
    """Run a task using SMCP tools and ask_human_cli"""
    
    # Enable standalone logging mode
    from .logging_setup import enable_standalone_mode
    enable_standalone_mode(browser_use_log_level="INFO")
    
    from .agents import Agent, AgentExecutor
    from .agents.models import CoreTool, ToolExecutorType
    from pathlib import Path
    
    async def run_task():
        # Load SMCP tools from registry if provided
        if smcp_registry:
            registry_path = Path(smcp_registry)
            if registry_path.exists():
                console.print(f"[blue]Loading SMCP tools from:[/] {smcp_registry}")
                # from_smcp_registry loads Agent with only SMCP tools from the registry
                agent = Agent.from_smcp_registry(str(registry_path))
                console.print(f"[blue]Loaded {len(agent.tools)} SMCP tools[/]")
            else:
                console.print(f"[yellow]Warning: Registry file not found:[/] {smcp_registry}")
                console.print(f"[yellow]Starting with no SMCP tools[/]")
                agent = Agent(description="", tools=[])
        else:
            console.print(f"[blue]No registry specified - running with core tools only[/]")
            agent = Agent(description="", tools=[])
        
        # Add ask_human_cli as a core tool to the agent
        agent.add_tool(CoreTool(name="ask_human_cli"))
        console.print(f"[blue]Added ask_human_cli to agent[/]")
        
        # Create executor with optional user_id for persistent browser profile
        if user_id:
            console.print(f"[blue]Using persistent browser profile for user:[/] {user_id}")
        executor = AgentExecutor(agent, user_id=user_id)
        
        try:
            console.print(f"\n[green]Running task...[/]")
            console.print(f"[dim]URL:[/] {url}")
            console.print(f"[dim]Task:[/] {prompt}")
            
            # Get SMCP tool names for display
            smcp_tool_names = [
                t.name for t in agent.tools 
                if hasattr(t, 'tool_executor_type') and t.tool_executor_type == ToolExecutorType.SMCP
            ]
            if smcp_tool_names:
                console.print(f"[dim]Available SMCP tools:[/] {', '.join(smcp_tool_names)}")
            console.print(f"[dim]Human-in-loop:[/] ask_human_cli enabled\n")
            
            # Run task
            result = await executor.run(prompt, mode="loop", initial_url=url)
            
            console.print(f"\n[green]✓ Task complete![/]")
            console.print(f"[dim]Result:[/] {result}\n")
            
        except Exception as e:
            console.print(f"\n[red]✗ Error during task execution:[/] {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            await executor.cleanup()
    
    # Run async task
    asyncio.run(run_task())


def main():
    """Main entry point for CLI."""
    return cli()  # Return Click's exit code


if __name__ == "__main__":
    main()
