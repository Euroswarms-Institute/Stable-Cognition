import torch
import torch.nn as nn
from typing import Dict, Tuple

from .config import GlobularConfig
from .utils import RMSNorm, FeedForward
from .field import GlobularAgentField


class GlobularReasoningBlock(nn.Module):
    """Drop-in block for transformer-like models."""

    def __init__(self, cfg: GlobularConfig):
        super().__init__()
        assert cfg.dim > 0

        self.cfg = cfg
        self.field = GlobularAgentField(cfg)

        self.x_norm = RMSNorm(cfg.dim)
        self.agent_norm = RMSNorm(cfg.agent_dim)

        self.q_proj = nn.Linear(cfg.dim, cfg.agent_dim, bias=False)
        self.k_proj = nn.Linear(cfg.agent_dim, cfg.agent_dim, bias=False)
        self.v_proj = nn.Linear(cfg.agent_dim, cfg.agent_dim, bias=False)
        self.o_proj = nn.Linear(cfg.agent_dim, cfg.dim, bias=False)

        self.ff = FeedForward(cfg.dim, hidden_mult=4, dropout=cfg.dropout)
        self.final_norm = RMSNorm(cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)

        self.head_dim = cfg.agent_dim // cfg.num_heads

        self.residual_gate = nn.Sequential(
            nn.Linear(cfg.dim * 2, cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim),
            nn.Sigmoid(),
        )

    def token_reads_agent_field(
        self,
        x: torch.Tensor,
        agents: torch.Tensor,
    ) -> torch.Tensor:
        """Let tokens attend to agent field."""
        cfg = self.cfg
        b, t, d = x.shape
        n = agents.shape[1]
        hd = cfg.agent_dim // cfg.num_heads

        q = self.q_proj(x).view(b, t, cfg.num_heads, hd).transpose(1, 2)
        k = self.k_proj(agents).view(b, n, cfg.num_heads, hd).transpose(1, 2)
        v = self.v_proj(agents).view(b, n, cfg.num_heads, hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / hd ** 0.5

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, cfg.agent_dim)
        return self.o_proj(out)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass."""
        residual = x
        x_normed = self.x_norm(x)

        agents, diagnostics = self.field(x_normed)
        agents = self.agent_norm(agents)

        field_update = self.token_reads_agent_field(x_normed, agents)

        gate = self.residual_gate(torch.cat([x, field_update], dim=-1))
        x = residual + gate * field_update

        x = x + self.ff(self.final_norm(x))

        return x, diagnostics