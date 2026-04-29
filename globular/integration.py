"""
Globular Integration for Hugging Face models (DeepSeek, Qwen, Llama, etc.)

Usage:
    from globular.integration import apply_globular_to_model
    
    # Wrap any HuggingFace model
    model = apply_globular_to_model(model, config)
    
    # Or use the wrapper
    from globular.integration import GlobularLMWrapper
    model = GlobularLMWrapper("Qwen/Qwen2-0.5B", insert_after_layer=10)
"""

import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass

from .config import GlobularConfig
from .block import GlobularReasoningBlock


@dataclass
class GlobularIntegrationConfig:
    """Config for integrating Globular into existing models."""
    
    # Where to insert globular blocks
    insert_after_layers: List[int] = None  # None = all layers, or specify [10, 20]
    
    # Agent config
    num_agents: int = 64
    agent_dim: int = 256
    steps: int = 3
    
    # Whether to replace or enhance
    replace_ffn: bool = False  # Replace FFN with globular reasoning
    enhance_attention: bool = True  # Add agent field to attention
    
    # Model dimension (auto-detected if None)
    model_dim: Optional[int] = None
    
    # Number of heads (auto-detected if None)
    num_heads: Optional[int] = None
    
    def __post_init__(self):
        if self.insert_after_layers is None:
            self.insert_after_layers = []  # Will be set based on model depth


class GlobularLayerWrapper(nn.Module):
    """Wrapper for a single transformer layer with Globular reasoning."""
    
    def __init__(
        self,
        original_layer: nn.Module,
        config: GlobularConfig,
        replace_ffn: bool = False,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.globular_block = GlobularReasoningBlock(config)
        self.replace_ffn = replace_ffn
        
    def forward(self, *args, **kwargs):
        # Standard forward through original layer
        output = self.original_layer(*args, **kwargs)
        
        # If output is a tuple (hidden_states,), apply globular
        if isinstance(output, tuple):
            hidden_states = output[0]
            states, diagnostics = self.globular_block(hidden_states)
            return (states,) + output[1:]
        else:
            # Apply globular directly
            hidden_states = output
            states, diagnostics = self.globular_block(hidden_states)
            return states


def apply_globular_to_model(
    model: nn.Module,
    integration_config: Optional[GlobularIntegrationConfig] = None,
) -> nn.Module:
    """
    Apply Globular reasoning to a Hugging Face model.
    
    Args:
        model: Any transformer model (Qwen, DeepSeek, Llama, etc.)
        integration_config: Configuration for integration
    
    Returns:
        Modified model with Globular layers inserted
    
    Example:
        from transformers import AutoModelForCausalLM
        import torch
        
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
        model = apply_globular_to_model(model)
        
        # Generate
        outputs = model(inputs)
    """
    
    if integration_config is None:
        integration_config = GlobularIntegrationConfig()
    
    config = GlobularConfig(
        dim=integration_config.model_dim or 512,
        num_agents=integration_config.num_agents,
        agent_dim=integration_config.agent_dim,
        steps=integration_config.steps,
        num_heads=integration_config.num_heads or 8,
    )
    
    # Find transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Qwen-style: model.model.layers
        layers = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "transformer"):
        # Llama-style: model.model.transformer.h
        layers = model.model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        # Mixtral-style
        layers = model.model.layers
    elif hasattr(model, "layers"):
        # Direct
        layers = model.layers
    else:
        # Try generic search
        layers = None
        for name, module in model.named_modules():
            if "layer" in name.lower() and isinstance(module, nn.ModuleList):
                layers = module
                break
        
        if layers is None:
            raise ValueError("Could not find transformer layers in model")
    
    num_layers = len(layers)
    
    # Determine insertion points
    insert_after = integration_config.insert_after_layers
    if not insert_after:
        # Insert after every layer
        insert_after = list(range(num_layers))
    
    # Create new layer list with globular
    new_layers = nn.ModuleList()
    
    for i, layer in enumerate(layers):
        new_layers.append(layer)
        
        if i in insert_after:
            # Insert globular block after this layer
            globular_block = GlobularReasoningBlock(config)
            
            # Create wrapper that can integrate
            wrapper = GlobularLayerWrapper(layer, config, integration_config.replace_ffn)
            
            # Store in model
            new_layers.append(globular_block)
    
    # Replace layers in model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
    elif hasattr(model, "layers"):
        model.layers = new_layers
    else:
        # Generic replacement
        for name, module in model.named_children():
            if isinstance(module, nn.ModuleList):
                setattr(model, name, new_layers)
                break
    
    return model


class GlobularLMWrapper(nn.Module):
    """Wrapper that adds Globular reasoning to any HF causal LM."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        insert_after_layer: int = 10,
        num_agents: int = 64,
        agent_dim: int = 256,
    ):
        super().__init__()
        
        # Try to load model
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            
            self.hidden_size = config.hidden_size
            self.num_attention_heads = getattr(config, "num_attention_heads", 8)
            
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except ImportError:
            # Fallback: create simple model
            model = None
            self.hidden_size = 512
            self.num_attention_heads = 8
        
        self.model_name = model_name
        self.insert_after_layer = insert_after_layer
        self.num_agents = num_agents
        self.agent_dim = agent_dim
        
        # Create globular config matching model
        gconfig = GlobularConfig(
            dim=self.hidden_size,
            num_agents=num_agents,
            agent_dim=agent_dim,
            steps=3,
            num_heads=self.num_attention_heads,
        )
        
        self.globular_block = GlobularReasoningBlock(gconfig)
        
        self.base_model = model
        
    def forward(self, *args, **kwargs):
        if self.base_model is None:
            # Dummy forward for testing
            input_ids = args[0] if args else kwargs.get("input_ids")
            b, t = input_ids.shape
            return torch.randn(b, t, self.hidden_size)
        
        # Forward through base model
        outputs = self.base_model(*args, **kwargs)
        
        # Apply globular to last hidden states
        if hasattr(outputs, "hidden_states"):
            hidden = outputs.hidden_states[-1]
            refined, _ = self.globular_block(hidden)
            outputs.hidden_states.append(refined)
        
        return outputs
    
    def generate(self, *args, **kwargs):
        if self.base_model:
            return self.base_model.generate(*args, **kwargs)
        raise NotImplementedError("Base model not loaded")


def add_globular_to_transformer(
    model: nn.Module,
    num_globular_layers: int = 2,
) -> nn.Module:
    """
    Add Globular reasoning layers to an existing transformer model.
    
    This inserts GlobularReasoningBlock layers at strategic positions
    to enhance reasoning without full model replacement.
    """
    
    # Detect model type and dimensions
    if hasattr(model, "config"):
        hidden_size = getattr(model.config, "hidden_size", 512)
        num_heads = getattr(model.config, "num_attention_heads", 8)
        num_layers = getattr(model.config, "num_hidden_layers", 12)
    else:
        hidden_size = 512
        num_heads = 8
        num_layers = 12
    
    gconfig = GlobularConfig(
        dim=hidden_size,
        num_agents=num_agents if (num_agents := 64) else 64,
        agent_dim=agent_dim if (agent_dim := 256) else 256,
        steps=3,
        num_heads=num_heads,
    )
    
    # Insert globular layers after every N layers
    insert_every = max(1, num_layers // num_globular_layers)
    insert_points = [i * insert_every for i in range(1, num_globular_layers + 1)]
    insert_points = [min(i, num_layers - 1) for i in insert_points]
    
    # Get the model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Could not find model layers")
    
    # Wrap with globular
    for idx in insert_points:
        original_layer = layers[idx]
        layers[idx] = GlobularLayerWrapper(original_layer, gconfig)
    
    return model


def create_globular_model(
    base_model: str = "Qwen/Qwen2-0.5B",
    enable_globular: bool = True,
    num_agents: int = 64,
    agent_dim: int = 256,
) -> nn.Module:
    """
    Create a model with optional Globular reasoning.
    
    Args:
        base_model: HuggingFace model name
        enable_globular: Whether to enable globular reasoning  
        num_agents: Number of agents per layer
        agent_dim: Agent embedding dimension
    
    Returns:
        Model with or without Globular
    """
    
    if not enable_globular:
        try:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(base_model)
        except:
            return GlobularLMWrapper(base_model)
    
    return GlobularLMWrapper(
        model_name=base_model,
        insert_after_layer=10,
        num_agents=num_agents,
        agent_dim=agent_dim,
    )