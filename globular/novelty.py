import torch
from typing import Tuple


class NoveltySearch:
    """Novelty search for exploration."""

    def __init__(
        self,
        archive_size: int = 500,
        k_nearest: int = 10,
        novelty_ratio_initial: float = 0.5,
        novelty_ratio_decay: float = 0.9,
        novelty_weight: float = 1.0,
    ):
        self.archive_size = archive_size
        self.k_nearest = k_nearest
        self.novelty_ratio_initial = novelty_ratio_initial
        self.novelty_ratio_decay = novelty_ratio_decay
        self.novelty_weight = novelty_weight

        self.behavior_archive = None
        self.archive_idx = 0
        self.archive_filled = torch.tensor(False)

    def init_archive(self, batch_size: int, agent_dim: int, device: torch.device):
        """Initialize empty archive."""
        self.behavior_archive = torch.zeros(
            batch_size, self.archive_size, agent_dim, device=device
        )
        self.archive_idx = 0
        self.archive_filled = torch.tensor(False, device=device)

    def compute_novelty(
        self,
        behaviors: torch.Tensor,
        archive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute novelty scores vs archive."""
        b, n, d = behaviors.shape

        if archive is None or self.archive_filled.sum() == 0:
            return torch.zeros(b, n, device=behaviors.device), torch.zeros(b, device=behaviors.device)

        normed_behaviors = torch.nn.functional.normalize(behaviors, dim=-1)
        normed_archive = torch.nn.functional.normalize(archive, dim=-1)

        similarities = torch.einsum("bnd,bsd->bns", normed_behaviors, normed_archive)

        k = min(self.k_nearest, self.archive_size)
        topk_sims, _ = torch.topk(similarities, k=k, dim=-1)
        novelty = 1.0 - topk_sims.mean(dim=-1)

        mean_novelty = novelty.mean(dim=-1)

        return novelty, mean_novelty

    def update_archive(self, behaviors: torch.Tensor):
        """Update archive with new behaviors."""
        if self.behavior_archive is None:
            return

        b = behaviors.size(0)

        idx = self.archive_idx
        self.behavior_archive[:, idx] = behaviors[:, idx % behaviors.size(1)]

        self.archive_idx = (self.archive_idx + 1) % self.archive_size

        if self.archive_idx == 0:
            self.archive_filled = torch.tensor(True)

    def decay_ratio(self, step: int) -> float:
        """Apply exponential decay to novelty ratio."""
        return self.novelty_ratio_initial * (self.novelty_ratio_decay ** step)

    def combined_selection(
        self,
        fitness: torch.Tensor,
        novelty: torch.Tensor,
        novelty_ratio: float,
    ) -> torch.Tensor:
        """Combine fitness + novelty for selection."""
        fit_norm = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-8)
        nov_norm = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-8)

        return fit_norm * (1.0 - novelty_ratio) + nov_norm * novelty_ratio * self.novelty_weight