from dataclasses import dataclass
from typing import Optional


@dataclass
class GlobularConfig:
    """Main configuration for Globular Reasoning Architecture."""

    dim: int

    # Agent population
    num_agents: int = 128
    agent_dim: int = 256
    steps: int = 4
    num_heads: int = 8

    # Dropout & regularization
    dropout: float = 0.1
    temperature: float = 0.75

    # Sparse attention
    use_sparse_topk: bool = True
    topk: int = 24

    # Core/Halo dynamics
    use_core_halo: bool = True
    core_ratio: float = 0.2

    # Death/Spawn
    spawn_rate: float = 0.05
    death_rate: float = 0.05

    # Residual & field dynamics
    residual_scale: float = 0.4
    field_scale: float = 0.3
    damping: float = 0.15

    # Gates
    use_energy_gate: bool = True
    normalize_agents: bool = True

    return_diagnostics: bool = True

    # Evolution (GA)
    use_evolution: bool = True
    evolution_trigger: str = "hybrid"

    mutation_rate_base: float = 0.015
    mutation_rate_adaptive: bool = True
    crossover_rate: float = 0.7
    crossover_alpha_min: float = 0.3
    crossover_alpha_max: float = 0.7

    truncation_ratio_min: float = 0.1
    truncation_ratio_max: float = 0.4
    truncation_adaptive: bool = True

    # Fitness sharing
    fitness_sharing_lambda: float = 1.0
    use_parameterized_similarity: bool = True

    # Species
    num_species: int = 4
    clustering_k: int = 4
    clustering_knn: int = 8

    # Coevolution
    use_coevolution: bool = True
    num_populations: int = 2
    coevolution_merge_step: int = 3

    # Meta-memory
    use_meta_memory: bool = True
    meta_memory_size: int = 4

    # Bottleneck adaptive
    use_bottleneck_adaptive: bool = True
    base_field_steps: int = 4
    base_ga_steps: int = 2
    budget_factor: float = 0.8

    # Novelty search
    use_novelty_search: bool = True
    novelty_ratio_initial: float = 0.4
    novelty_ratio_decay: float = 0.9
    archive_size: int = 500
    k_nearest: int = 10
    novelty_weight: float = 0.8

    # Memetic
    use_memetic: bool = True
    local_search_steps: int = 2
    local_learning_rate: float = 0.005
    local_objective: str = "combined"

    # Hierarchical
    use_hierarchical: bool = True
    sub_species_per_species: int = 2
    sub_species_distance_threshold: float = 0.3

    # Optimizations
    use_gqa: bool = True
    kv_heads: int = 4
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_weight_tying: bool = True
    use_loop_weight_sharing: bool = True
    use_mixed_precision: bool = True

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.agent_dim <= 0:
            raise ValueError("agent_dim must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.agent_dim % self.num_heads != 0:
            raise ValueError("agent_dim must be divisible by num_heads")
        if self.kv_heads <= 0:
            raise ValueError("kv_heads must be positive")
        if self.use_gqa and self.num_heads % self.kv_heads != 0:
            # Keep grouped-query attention valid without surprising runtime shape errors.
            for candidate in (64, 32, 16, 8, 4, 2, 1):
                if self.num_heads % candidate == 0:
                    self.kv_heads = candidate
                    break
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if not 0 < self.core_ratio <= 1:
            raise ValueError("core_ratio must be in (0, 1]")


@dataclass
class CPUOptimizedConfig:
    """Lightweight config for CPU inference."""

    dim: int = 512
    num_agents: int = 64
    agent_dim: int = 256
    steps: int = 3
    num_heads: int = 4
    kv_heads: int = 2

    dropout: float = 0.0
    temperature: float = 1.0

    use_sparse_topk: bool = True
    topk: int = 16

    use_core_halo: bool = False
    core_ratio: float = 0.3

    spawn_rate: float = 0.0
    death_rate: float = 0.0

    residual_scale: float = 0.3
    field_scale: float = 0.2
    damping: float = 0.2

    use_energy_gate: bool = False
    normalize_agents: bool = True
    return_diagnostics: bool = False

    use_evolution: bool = False
    use_novelty_search: bool = False
    use_memetic: bool = False
    use_hierarchical: bool = False
    use_coevolution: bool = False
    use_meta_memory: bool = False
    meta_memory_size: int = 1
    num_species: int = 1
    sub_species_per_species: int = 1
    use_bottleneck_adaptive: bool = False
    use_gqa: bool = True
    use_parameterized_similarity: bool = False
    use_gradient_checkpointing: bool = False
    use_weight_tying: bool = False
    use_loop_weight_sharing: bool = False
    use_mixed_precision: bool = False

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.agent_dim <= 0:
            raise ValueError("agent_dim must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.agent_dim % self.num_heads != 0:
            raise ValueError("agent_dim must be divisible by num_heads")
        if self.use_gqa and self.num_heads % self.kv_heads != 0:
            for candidate in (16, 8, 4, 2, 1):
                if self.num_heads % candidate == 0:
                    self.kv_heads = candidate
                    break
