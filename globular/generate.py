#!/usr/bin/env python3
"""
Globular Generate - Test generation with integrated models
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from globular.hf_utils import safe_load_causal_lm, safe_load_tokenizer, ensure_tokenizer_padding


def _model_device(model):
    return next(model.parameters()).device


def generate(model_path, prompt, max_tokens, temperature, top_p):
    """Generate text with integrated model"""
    print(f"Loading {model_path}...")
    
    model = safe_load_causal_lm(
        model_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = ensure_tokenizer_padding(safe_load_tokenizer(model_path))
    
    print(f"Model loaded. Device: {_model_device(model)}")
    print(f"\nPrompt: {prompt}")
    print("Generating...")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(_model_device(model)) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n{'='*60}")
    print(f"Output:")
    print(f"{'='*60}")
    print(result)
    print(f"{'='*60}")
    
    return result


def chat_mode(model_path):
    """Interactive chat mode"""
    print(f"Loading {model_path}...")
    
    model = safe_load_causal_lm(
        model_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = ensure_tokenizer_padding(safe_load_tokenizer(model_path))
    
    print(f"\n[Chat mode] Type 'quit' to exit\n")
    
    while True:
        prompt = input("> ")
        if prompt.lower() == "quit":
            break
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(_model_device(model)) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove input prompt from output
        if result.startswith(prompt):
            result = result[len(prompt):].strip()
        
        print(f"\n{result}\n")


def main():
    parser = argparse.ArgumentParser(description="Globular Generate")
    
    parser.add_argument("model", nargs="?", default="./output/globular-integrated", help="Model path")
    parser.add_argument("--prompt", "-p", default="The capital of France is", help="Prompt")
    parser.add_argument("--max-tokens", "-m", default=50, help="Max new tokens")
    parser.add_argument("--temperature", "-t", default=0.7, help="Temperature")
    parser.add_argument("--top-p", default=0.9, help="Top-p sampling")
    parser.add_argument("--chat", "-c", action="store_true", help="Chat mode")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.chat or args.interactive:
        chat_mode(args.model)
    else:
        generate(args.model, args.prompt, args.max_tokens, args.temperature, args.top_p)


if __name__ == "__main__":
    main()
