#!/usr/bin/env python3
"""Rich-based terminal launcher for Globular tools."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(force_terminal=False)


def star_growth_chart():
    """Display an ASCII-safe scaling chart."""
    from globular.cli import auto_scale

    data = [(16, 3), (32, 10), (64, 30), (128, 80), (256, 180), (512, 450), (1024, 1000)]

    table = Table(title="Star Growth Chart (params in millions)", show_header=True, header_style="bold cyan")
    table.add_column("Agents", style="cyan", justify="right")
    table.add_column("Params (M)", style="green", justify="right")
    table.add_column("Visual", style="yellow")

    width = 40
    for agents, params in data:
        filled = min(width, max(1, int(params / 25)))
        visual = "#" * filled + "-" * (width - filled)
        table.add_row(f"{agents:,}", f"{params:,}M", visual)

    console.print(table)

    table2 = Table(title="Auto-Scaling Examples", show_header=True, header_style="bold cyan")
    table2.add_column("Agents", style="cyan", justify="right")
    table2.add_column("Auto Dim", style="yellow", justify="right")
    table2.add_column("Steps", style="green", justify="right")
    table2.add_column("Heads", style="magenta", justify="right")

    for n in [64, 128, 256, 512, 1024, 4096, 16384, 64000]:
        auto = auto_scale(n)
        table2.add_row(f"{n:,}", f"{auto['agent_dim']:,}", str(auto["steps"]), str(auto["num_heads"]))

    console.print(table2)


def show_help():
    table = Table(title="Globular Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    table.add_column("Example", style="dim")

    for cmd, desc, example in [
        ("python -m globular.cli", "Interactive CLI menu", "python -m globular.cli"),
        ("python -m globular.train", "Training with resume/metrics", "python -m globular.train --agents 128"),
        ("python -m globular.generate", "Text generation", "python -m globular.generate ./model --chat"),
        ("python -m globular.hub", "Download/list models", "python -m globular.hub --list"),
    ]:
        table.add_row(cmd, desc, example)

    console.print(table)


def run_command(module: str, args: list[str] | None = None):
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    subprocess.run(cmd, check=False)


def main():
    console.print(
        Panel.fit(
            "GLOBULAR v0.2.0\n"
            "Reasoning Architecture for Language Models\n"
            "Run: python -m globular.cli for full interactive mode",
            border_style="cyan",
        )
    )

    table = Table(box=None, pad_edge=True)
    table.add_column("", style="cyan")
    table.add_column("Option")
    table.add_column("Description", style="dim")

    options = [
        ("1", "Star Growth Chart", "View scaling visualization"),
        ("2", "CLI (Interactive)", "Full menu system"),
        ("3", "Training", "Training help"),
        ("4", "Generation", "Generation help"),
        ("5", "Model Hub", "List downloadable models"),
        ("6", "Help", "All commands"),
    ]
    for key, name, desc in options:
        table.add_row(f"[{key}]", name, desc)
    console.print(table)


if __name__ == "__main__":
    main()
    print("\nRun 'python -m globular.cli' for full interactive mode.")
