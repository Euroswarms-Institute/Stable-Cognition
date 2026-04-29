import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .config import GlobularConfig
from .utils import RMSNorm, FeedForward
from .attention import SparseAttention, GQA
from .evolution import EvolutionOperators
from .novelty import NoveltySearch


class GlobularAgentField(nn.Module):
    """
    Evolves a population of latent hypothesis-agents.

    Agents are latent hypothesis carriers: Tensor[B, N, Da]
    N = number of agents, Da = agent dimension
    """

    def __init__(self, cfg: GlobularConfig):
        super().__init__()
        if cfg.agent_dim % cfg.num_heads != 0:
            raise ValueError("cfg.agent_dim must be divisible by cfg.num_heads")

        self.cfg = cfg
        self.head_dim = cfg.agent_dim // cfg.num_heads

        self.agent_seed = nn.Parameter(torch.randn(cfg.num_agents, cfg.agent_dim) * 0.02)

        self.input_to_agents = nn.Linear(cfg.dim, cfg.agent_dim)
        self.agents_to_model = nn.Linear(cfg.agent_dim, cfg.dim)

        self.agent_norm = RMSNorm(cfg.agent_dim)
        self.context_norm = RMSNorm(cfg.agent_dim)

        kv_heads = cfg.kv_heads if cfg.use_gqa else cfg.num_heads
        if cfg.use_gqa and cfg.num_heads % kv_heads != 0:
            raise ValueError("cfg.kv_heads must divide cfg.num_heads when use_gqa=True")

        self.q_proj = nn.Linear(cfg.agent_dim, cfg.agent_dim, bias=False)
        self.k_proj = nn.Linear(cfg.agent_dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.agent_dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.agent_dim, cfg.agent_dim, bias=False)

        self.input_q = nn.Linear(cfg.agent_dim, cfg.agent_dim, bias=False)
        self.input_k = nn.Linear(cfg.agent_dim, cfg.agent_dim, bias=False)
        self.input_v = nn.Linear(cfg.agent_dim, cfg.agent_dim, bias=False)

        self.energy_net = nn.Sequential(
            nn.Linear(cfg.agent_dim, cfg.agent_dim),
            nn.GELU(),
            nn.Linear(cfg.agent_dim, 1),
        )

        self.update_gate = nn.Sequential(
            nn.Linear(cfg.agent_dim * 3, cfg.agent_dim),
            nn.GELU(),
            nn.Linear(cfg.agent_dim, cfg.agent_dim),
            nn.Sigmoid(),
        )

        self.agent_ff = FeedForward(cfg.agent_dim, hidden_mult=4, dropout=cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

        self.final_norm = RMSNorm(cfg.agent_dim)

        if cfg.use_parameterized_similarity:
            self.similarity_net = nn.Sequential(
                nn.Linear(cfg.agent_dim * 2, cfg.agent_dim),
                nn.GELU(),
                nn.Linear(cfg.agent_dim, cfg.agent_dim),
                nn.GELU(),
                nn.Linear(cfg.agent_dim, 1),
            )

        evolution_hidden = max(1, cfg.agent_dim // 2)
        self.evolution_gate = nn.Sequential(
            nn.Linear(cfg.agent_dim + 3, evolution_hidden),
            nn.GELU(),
            nn.Linear(evolution_hidden, 1),
            nn.Sigmoid(),
        )

        self.meta_memory_init = nn.Parameter(
            torch.randn(cfg.num_agents, cfg.meta_memory_size) * 0.01
        )
        self.mutation_rate_net = nn.Sequential(
            nn.Linear(cfg.agent_dim + cfg.meta_memory_size, cfg.agent_dim),
            nn.GELU(),
            nn.Linear(cfg.agent_dim, 1),
            nn.Sigmoid(),
        )

        self.species_centroids = nn.Parameter(
            torch.randn(cfg.num_species, cfg.agent_dim) * 0.02
        )

        if cfg.use_coevolution:
            self.explorer_seed = nn.Parameter(
                torch.randn(cfg.num_agents, cfg.agent_dim) * 0.02
            )
            self.exploiter_seed = nn.Parameter(
                torch.randn(cfg.num_agents, cfg.agent_dim) * 0.02
            )
            self.explorer_energy_net = nn.Sequential(
                nn.Linear(cfg.agent_dim, cfg.agent_dim),
                nn.GELU(),
                nn.Linear(cfg.agent_dim, 1),
            )
            self.exploiter_energy_net = nn.Sequential(
                nn.Linear(cfg.agent_dim, cfg.agent_dim),
                nn.GELU(),
                nn.Linear(cfg.agent_dim, 1),
            )

        if cfg.use_hierarchical:
            self.sub_species_centroids = nn.Parameter(
                torch.randn(
                    cfg.num_species * cfg.sub_species_per_species, cfg.agent_dim
                )
                * 0.02
            )

        self.sparse_attn = SparseAttention(cfg.num_heads, self.head_dim)

        self.novelty_search = None
        if cfg.use_novelty_search:
            self.novelty_search = NoveltySearch(
                archive_size=cfg.archive_size,
                k_nearest=cfg.k_nearest,
                novelty_ratio_initial=cfg.novelty_ratio_initial,
                novelty_ratio_decay=cfg.novelty_ratio_decay,
                novelty_weight=cfg.novelty_weight,
            )

    def _agent_interaction(self, agents: torch.Tensor) -> torch.Tensor:
        """Agent-to-agent attention."""
        cfg = self.cfg
        b, n, d = agents.shape

        q = self.q_proj(agents)
        k = self.k_proj(agents)
        v = self.v_proj(agents)

        h = cfg.num_heads
        hd = self.head_dim
        kv = cfg.kv_heads if cfg.use_gqa else h

        q = q.view(b, n, h, hd).transpose(1, 2)
        k = k.view(b, n, kv, hd).transpose(1, 2)
        v = v.view(b, n, kv, hd).transpose(1, 2)

        if cfg.use_gqa and kv < h:
            k = k.repeat_interleave(h // kv, dim=1)
            v = v.repeat_interleave(h // kv, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(hd)
        scores = scores / max(cfg.temperature, 1e-6)

        if cfg.use_sparse_topk:
            attn = self.sparse_attn(scores, min(cfg.topk, cfg.num_agents))
        else:
            attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.o_proj(out)

    def _input_influence(self, agents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Agents attend to context."""
        cfg = self.cfg
        b, n, d = agents.shape
        tc = context.shape[1]

        q = self.input_q(agents).view(b, n, -1, self.head_dim).transpose(1, 2)
        k = self.input_k(context).view(b, tc, -1, self.head_dim).transpose(1, 2)
        v = self.input_v(context).view(b, tc, -1, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return out

    def _energy(self, agents: torch.Tensor) -> torch.Tensor:
        """Compute energy (lower = better hypothesis)."""
        return self.energy_net(agents).squeeze(-1)

    def _core_halo_mix(
        self,
        agents: torch.Tensor,
        energy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Form core/halo split."""
        cfg = self.cfg
        b, n, d = agents.shape

        core_k = max(1, int(n * cfg.core_ratio))
        confidence = torch.softmax(-energy, dim=-1)

        _, core_idx = torch.topk(confidence, k=core_k, dim=-1)
        gather_idx = core_idx.unsqueeze(-1).expand(-1, -1, d)
        core_agents = torch.gather(agents, dim=1, index=gather_idx)

        core_centroid = core_agents.mean(dim=1, keepdim=True)
        attraction = core_centroid - agents
        agents = agents + cfg.field_scale * attraction * confidence.unsqueeze(-1)

        return agents, confidence

    def initialize_agents(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize agents from input."""
        b = x.size(0)

        pooled = x.mean(dim=1)
        pooled = self.input_to_agents(pooled)

        seed = self.agent_seed.unsqueeze(0).expand(b, -1, -1)
        agents = seed + pooled.unsqueeze(1)

        meta_memory = self.meta_memory_init.unsqueeze(0).expand(b, -1, -1)

        if self.cfg.normalize_agents:
            agents = F.normalize(agents, dim=-1)

        return agents, meta_memory

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through agent field."""
        cfg = self.cfg

        context = self.context_norm(self.input_to_agents(x))
        agents, meta_memory = self.initialize_agents(x)

        energy_history = []
        diversity_history = []

        for step in range(cfg.steps):
            old_agents = agents
            agents_normed = self.agent_norm(agents)

            peer = self._agent_interaction(agents_normed)
            inp = self._input_influence(agents_normed, context)

            gate = self.update_gate(torch.cat([agents_normed, peer, inp], dim=-1))
            update = gate * peer + (1.0 - gate) * inp
            agents = agents + cfg.residual_scale * update

            agents = agents + self.agent_ff(self.agent_norm(agents))

            energy = self._energy(agents)

            if cfg.use_core_halo:
                agents, confidence = self._core_halo_mix(agents, energy)
            else:
                confidence = torch.softmax(-energy, dim=-1)

            if cfg.use_energy_gate:
                agents = agents * confidence.unsqueeze(-1) + old_agents * (
                    1.0 - confidence.unsqueeze(-1)
                )

            agents = (1.0 - cfg.damping) * agents + cfg.damping * old_agents

            if cfg.normalize_agents:
                agents = F.normalize(agents, dim=-1)

            with torch.no_grad():
                energy_history.append(energy.mean(dim=-1))
                centered = agents - agents.mean(dim=1, keepdim=True)
                diversity = centered.pow(2).mean(dim=(1, 2))
                diversity_history.append(diversity)

        agents = self.final_norm(agents)
        final_energy = self._energy(agents)
        confidence = torch.softmax(-final_energy, dim=-1)

        diagnostics = {}
        if cfg.return_diagnostics:
            diagnostics = {
                "agent_energy": final_energy.detach(),
                "agent_confidence": confidence.detach(),
                "mean_energy_history": torch.stack(energy_history, dim=1).detach(),
                "diversity_history": torch.stack(diversity_history, dim=1).detach(),
            }

        return agents, diagnostics
