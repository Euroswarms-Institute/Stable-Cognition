#!/usr/bin/env python3
"""
Globular Model Hub - Download and manage pretrained models
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from globular.hf_utils import safe_load_causal_lm, safe_load_tokenizer, ensure_tokenizer_padding

console = Console()


def _short(text: str, limit: int = 36) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


MODELS = {
    "qwen2-0.5b": {
        "name": "Qwen/Qwen2-0.5B",
        "desc": "Small efficient model",
        "size": "~900MB",
        "hidden": 896,
        "layers": 24,
    },
    "qwen2-1.5b": {
        "name": "Qwen/Qwen2-1.5B", 
        "desc": "Medium model",
        "size": "~1.5GB",
        "hidden": 1536,
        "layers": 28,
    },
    "qwen2-7b": {
        "name": "Qwen/Qwen2-7B",
        "desc": "Large model (requires GPU)",
        "size": "~7GB",
        "hidden": 3584,
        "layers": 28,
    },
    "llama2-7b": {
        "name": "meta-llama/Llama-2-7b",
        "desc": "Llama 2 7B",
        "size": "~7GB",
        "hidden": 4096,
        "layers": 32,
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "desc": "Mistral 7B",
        "size": "~7GB",
        "hidden": 4096,
        "layers": 32,
    },
    "deepseek-7b": {
        "name": "deepseek-ai/DeepSeek-Coder-7B",
        "desc": "DeepSeek Coder 7B",
        "size": "~7GB",
        "hidden": 4096,
        "layers": 30,
    },
}


def list_models():
    """List available models"""
    table = Table(title="Available Models", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="cyan")
    table.add_column("Model", style="white", no_wrap=True)
    table.add_column("Size", style="yellow")
    table.add_column("Hidden", style="green")
    table.add_column("Description", style="dim")
    
    for key, m in MODELS.items():
        table.add_row(key, _short(m["name"]), m["size"], str(m["hidden"]), m["desc"])
    
    console.print(table)


def download_model(key, output_dir=None):
    """Download a model"""
    if key not in MODELS:
        console.print(f"[red]Unknown model: {key}[/red]")
        return
    
    m = MODELS[key]
    model_name = m["name"]
    
    if output_dir is None:
        output_dir = f"./models/{key}"
    
    console.print(f"[cyan]Downloading {model_name}...[/cyan]")
    console.print(f"[dim]This may take a while...[/dim]")
    
    try:
        console.print("[dim]Loading tokenizer...[/dim]")
        tokenizer = ensure_tokenizer_padding(safe_load_tokenizer(model_name))
        
        console.print("[dim]Loading model (this may take a while)...[/dim]")
        model = safe_load_causal_lm(
            model_name,
            device_map="cpu",  # Always CPU for download
            torch_dtype="auto",
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        console.print(f"[dim]Saving to {output_dir}...[/dim]")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        console.print(Panel.fit(
            f"[green]Download complete![/green]\n"
            f"Model: {model_name}\n"
            f"Saved: {output_dir}",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Globular Model Hub")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--download", "-d", help="Download model by key")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--all", "-a", action="store_true", help="Download all models")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.download:
        download_model(args.download, args.output)
    elif args.all:
        for key in MODELS:
            download_model(key, f"./models/{key}")
    else:
        console.print("[bold]Globular Model Hub[/bold]\n")
        console.print("Options:")
        console.print("  --list, -l      List models")
        console.print("  --download KEY  Download model")
        console.print("  --all, -a       Download all")
        console.print("\nExample:")
        console.print("  python -m globular.hub --list")
        console.print("  python -m globular.hub --download qwen2-0.5b")


if __name__ == "__main__":
    main()
