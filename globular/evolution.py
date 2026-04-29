import torch
import torch.nn.functional as F
from typing import Tuple, Dict


class EvolutionOperators:
    """Genetic algorithm operators for agent evolution."""

    @staticmethod
    def selection_truncation(
        agents: torch.Tensor,
        fitness: torch.Tensor,
        truncation_ratio: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Truncation selection: keep top-k, replace rest."""
        b, n, d = agents.shape
        keep = max(2, int(n * truncation_ratio))

        sorted_fit, indices = torch.sort(fitness, dim=-1, descending=True)
        top_indices = indices[:, :keep]

        batch_idx = torch.arange(b, device=agents.device).unsqueeze(-1).expand(-1, keep)
        selected = agents[batch_idx, top_indices]

        return selected, top_indices

    @staticmethod
    def crossover_blx_alpha(
        parents1: torch.Tensor,
        parents2: torch.Tensor,
        alpha_min: float = 0.3,
        alpha_max: float = 0.7,
    ) -> torch.Tensor:
        """BLX-alpha crossover."""
        b, n, d = parents1.shape

        min_vals = torch.minimum(parents1, parents2)
        max_vals = torch.maximum(parents1, parents2)
        range_vals = max_vals - min_vals

        child1 = parents1 + alpha_min * range_vals * torch.rand_like(parents1) * 2 - alpha_min * range_vals
        child2 = parents2 + alpha_max * range_vals * torch.rand_like(parents2) * 2 - alpha_max * range_vals

        return (child1 + child2) / 2.0

    @staticmethod
    def mutation_gaussian(
        agents: torch.Tensor,
        mutation_rate: float = 0.02,
    ) -> torch.Tensor:
        """Gaussian mutation."""
        noise = torch.randn_like(agents)
        return agents + noise * mutation_rate

    @staticmethod
    def compute_fitness(
        energy: torch.Tensor,
        similarity: torch.Tensor = None,
        lambda_penalty: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute composite fitness."""
        confidence = torch.softmax(-energy, dim=-1)

        if similarity is not None:
            avg_sim = similarity.mean(dim=-1, keepdim=True)
            fitness = confidence * torch.exp(-lambda_penalty * avg_sim)
        else:
            fitness = confidence

        return fitness, confidence

    @staticmethod
    def species_clustering(
        agents: torch.Tensor,
        centroids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cluster agents by nearest centroid."""
        normed_agents = F.normalize(agents, dim=-1)
        normed_centroids = F.normalize(centroids, dim=-1)

        similarities = torch.einsum("bnd,cd->bc", normed_agents, normed_centroids)
        species_ids = similarities.argmax(dim=-1)

        species_mask = F.one_hot(species_ids, num_classes=centroids.size(0)).float()

        return species_ids, species_mask

    @staticmethod
    def fitness_sharing(
        fitness: torch.Tensor,
        similarity: torch.Tensor,
        lambda_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Apply fitness sharingpenalty."""
        penalty = 1.0 + lambda_penalty * similarity.sum(dim=-1, keepdim=True)
        return fitness / penalty