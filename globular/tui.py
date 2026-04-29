#!/usr/bin/env python3
"""
Globular TUI - Rich-based Terminal UI Launcher
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def star_growth_chart():
    """Display star growth chart"""
    from globular.cli import auto_scale
    
    # Base chart
    data = [(16, 3), (32, 10), (64, 30), (128, 80), (256, 180), (512, 450), (1024, 1000)]
    
    table = Table(title="Star Growth Chart (params in millions)", show_header=True, header_style="bold cyan")
    table.add_column("Agents", style="cyan", justify="right")
    table.add_column("Params (M)", style="green", justify="right")
    table.add_column("Visual", style="yellow")
    
    for agents, params in data:
        visual = "█" * int(params / 25) + "░" * (40 - int(params / 25))
        table.add_row(f"{agents:,}", f"{params:,}M", visual)
    
    console.print(table)
    
    # Custom scaling
    console.print("\nCustom Scaling Examples:")
    table2 = Table(show_header=False)
    table2.add_column("Agents", style="cyan", justify="right")
    table2.add_column("Auto Dim", style="yellow", justify="right")  
    table2.add_column("Steps", style="green", justify="right")
    
    for n in [64, 128, 256, 512, 1024, 4096, 16384, 64000]:
        auto = auto_scale(n)
        table2.add_row(f"{n:,}", f"{auto['agent_dim']:,}", f"{auto['steps']}")
    
    console.print(table2)


def show_help():
    """Show all help options"""
    console.print("\n=== GLOBULAR COMMANDS ===\n")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Example", style="dim")
    
    cmds = [
        ("python -m globular.cli", "Interactive CLI menu", "python -m globular.cli"),
        ("python -m globular.train", "Training with auto-scale", "python -m globular.train --agents 128"),
        ("python -m globular.generate", "Text generation", "python -m globular.generate ./model --chat"),
        ("python -m globular.hub", "Download models", "python -m globular.hub --list"),
    ]
    
    for cmd, desc, ex in cmds:
        table.add_row(cmd, desc, ex)
    
    console.print(table)


def run_command(module, args=None):
    """Run a command"""
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def main():
    # Header
    console.print(Panel.fit(
        "GLOBULAR v0.2.0\n"
        "Reasoning Architecture for Language Models\n"
        "Run: python -m globular.cli for full interactive mode",
        border_style="cyan",
    ))
    
    # Menu
    options = [
        ("1", "Star Growth Chart", "View scaling visualization", star_growth_chart),
        ("2", "CLI (Interactive)", "Full menu system", lambda: run_command("globular.cli")),
        ("3", "Training", "Start training", lambda: run_command("globular.train", ["--help"])),
        ("4", "Generation", "Generate text", lambda: run_command("globular.generate", ["--help"])),
        ("5", "Model Hub", "Download models", lambda: run_command("globular.hub", ["--list"])),
        ("6", "Help", "All commands", show_help),
        ("Q", "Quit", "Exit", None),
    ]
    
    table = Table(box=None, pad_edge=True)
    table.add_column("", style="cyan")
    table.add_column("Option", style="white")
    table.add_column("Description", style="dim")
    
    for key, name, desc, _ in options:
        table.add_row(f"[{key}]", name, desc)
    
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
    print("\nRun 'python -m globular.cli' for full interactive mode.")