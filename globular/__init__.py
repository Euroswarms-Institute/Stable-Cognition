"""
Globular Reasoning Architecture

A PyTorch implementation inspired by globular clusters in space.
"""

from .config import GlobularConfig, CPUOptimizedConfig
from .utils import RMSNorm, FeedForward, count_parameters, get_device
from .attention import SparseAttention, GQA, MultiHeadAttention
from .evolution import EvolutionOperators
from .novelty import NoveltySearch
from .field import GlobularAgentField
from .block import GlobularReasoningBlock
from .model import (
    GlobularReasoningAdapter,
    ExampleTinyModelWithGlobularReasoning,
    CPUOptimizedModel,
)
from .integration import (
    GlobularIntegrationConfig,
    apply_globular_to_model,
    GlobularLMWrapper,
    add_globular_to_transformer,
    create_globular_model,
)

__version__ = "0.1.0"
__author__ = "Globular Team"

__all__ = [
    # Config
    "GlobularConfig",
    "CPUOptimizedConfig",
    # Utils
    "RMSNorm",
    "FeedForward",
    "count_parameters",
    "get_device",
    # Attention
    "SparseAttention",
    "GQA",
    "MultiHeadAttention",
    # Evolution
    "EvolutionOperators",
    # Novelty
    "NoveltySearch",
    # Core
    "GlobularAgentField",
    "GlobularReasoningBlock",
    # Models
    "GlobularReasoningAdapter",
    "ExampleTinyModelWithGlobularReasoning",
    "CPUOptimizedModel",
    # Integration
    "GlobularIntegrationConfig",
    "apply_globular_to_model",
    "GlobularLMWrapper",
    "add_globular_to_transformer",
    "create_globular_model",
]