import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAttention(nn.Module):
    """Top-k sparse attention."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, scores: torch.Tensor, topk: int) -> torch.Tensor:
        """
        scores: [B, H, N, N]
        Returns: [B, H, N, N] attention with top-k
        """
        if topk >= scores.size(-1):
            return F.softmax(scores, dim=-1)

        vals, idx = torch.topk(scores, k=topk, dim=-1)
        masked = torch.full_like(scores, float("-inf"))
        masked.scatter_(-1, idx, vals)
        return F.softmax(masked, dim=-1)


class GQA(nn.Module):
    """Grouped Query Attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_heads: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads or max(1, num_heads // 4)
        if self.kv_heads <= 0 or num_heads % self.kv_heads != 0:
            raise ValueError("kv_heads must be positive and divide num_heads")
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sparse_attn: SparseAttention = None,
        topk: int = None,
    ) -> torch.Tensor:
        """Forward pass with optional GQA."""
        b, n, d = x.shape

        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.kv_heads
            k = k.repeat(1, repeat_factor, 1, 1)
            v = v.repeat(1, repeat_factor, 1, 1)

        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)

        if sparse_attn and topk:
            attn = sparse_attn(scores, topk)
        else:
            attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.o_proj(out)


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape

        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.o_proj(out)
