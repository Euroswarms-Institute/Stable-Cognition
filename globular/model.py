import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .config import GlobularConfig, CPUOptimizedConfig
from .utils import RMSNorm, FeedForward, count_parameters
from .block import GlobularReasoningBlock


class GlobularReasoningAdapter(nn.Module):
    """Generic adapter for existing models."""

    def __init__(self, dim: int, num_agents: int = 128, agent_dim: int = 256):
        super().__init__()
        cfg = GlobularConfig(dim=dim, num_agents=num_agents, agent_dim=agent_dim)
        self.block = GlobularReasoningBlock(cfg)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out, _ = self.block(hidden_states)
        return out


class ExampleTinyModelWithGlobularReasoning(nn.Module):
    """Minimal example language model with Globular reasoning."""

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 512,
        depth: int = 4,
        max_seq_len: int = 1024,
        num_agents: int = 128,
        agent_dim: int = 256,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm1": RMSNorm(dim),
                        "attn": nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=8,
                            batch_first=True,
                            dropout=0.05,
                        ),
                        "globular": GlobularReasoningBlock(
                            GlobularConfig(
                                dim=dim,
                                num_agents=num_agents,
                                agent_dim=agent_dim,
                                steps=4,
                                num_heads=8,
                            )
                        ),
                        "norm2": RMSNorm(dim),
                        "ff": FeedForward(dim, hidden_mult=4, dropout=0.05),
                    }
                )
            )

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        b, t = input_ids.shape

        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :t, :]

        diagnostics = []

        causal_mask = torch.triu(
            torch.ones(t, t, device=input_ids.device, dtype=torch.bool), diagonal=1
        )

        for layer in self.layers:
            residual = x
            attn_out, _ = layer["attn"](
                layer["norm1"](x),
                layer["norm1"](x),
                layer["norm1"](x),
                attn_mask=causal_mask,
                need_weights=False,
            )
            x = residual + attn_out

            x, diag = layer["globular"](x)
            diagnostics.append(diag)

            x = x + layer["ff"](layer["norm2"](x))

        logits = self.lm_head(self.norm(x))

        if return_diagnostics:
            return logits, diagnostics

        return logits


class CPUOptimizedModel(nn.Module):
    """CPU-optimized model for inference."""

    def __init__(
        self,
        vocab_size: int = 16384,
        dim: int = 512,
        depth: int = 4,
        max_seq_len: int = 256,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm1": RMSNorm(dim),
                        "attn": nn.MultiheadAttention(
                            embed_dim=dim, num_heads=4, batch_first=True
                        ),
                        "globular": GlobularReasoningBlock(CPUOptimizedConfig()),
                        "norm2": RMSNorm(dim),
                        "ff": FeedForward(dim, hidden_mult=4),
                    }
                )
            )

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        b, t = input_ids.shape

        x = self.token_emb(input_ids).float()
        x = x + self.pos_emb[:, :t, :]

        diagnostics = []

        causal_mask = torch.triu(
            torch.ones(t, t, device=input_ids.device, dtype=torch.bool), diagonal=1
        )

        for layer in self.layers:
            residual = x
            attn_out, _ = layer["attn"](
                layer["norm1"](x),
                layer["norm1"](x),
                layer["norm1"](x),
                attn_mask=causal_mask,
                need_weights=False,
            )
            x = residual + attn_out

            x, diag = layer["globular"](x)
            diagnostics.append(diag)

            x = x + layer["ff"](layer["norm2"](x))

        logits = self.lm_head(self.norm(x))

        if return_diagnostics:
            return logits, diagnostics

        return logits