# Globular Reasoning Architecture

PyTorch implementation of a theoretical Reasoning Architecture inspired by globular clusters in space.

## Core Concept

A population of latent hypothesis-agents evolves in parallel through attention-like field dynamics. Stable clusters become attractors, and the resulting consensus field is projected back into model state.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# GPU training
python main.py

# CPU inference
python main.py cpu
```

## Model Variants

| Model | Params | Features | Hardware |
|-------|--------|----------|----------|
| Small | ~26M | Full evolution | 3050 |
| Large | ~322M | Full features | T4 |
| CPU | ~30-50M | Simplified | CPU |

## Architecture Features

- **Agent Field** - Population of latent hypothesis-agents
- **Core/Halo Dynamics** - Emergent specialization
- **Evolution (GA)** - Genetic algorithm selection
- **Species Formation** - Hierarchical clustering
- **Novelty Search** - Exploration vs exploitation
- **Memetic Algorithm** - Local refinement
- **Coevolution** - Explorer/exploiter populations
- **GQA** - Grouped query attention for efficiency

## Usage

```python
from main import GlobularConfig, GlobularReasoningBlock, ExampleTinyModelWithGlobularReasoning

# Small model
model = ExampleTinyModelWithGlobularReasoning(
    vocab_size=8192,
    dim=256,
    depth=2,
    num_agents=64,
    agent_dim=256,
)

# Forward pass
logits = model(input_ids)

# With diagnostics
logits, diagnostics = model(input_ids, return_diagnostics=True)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = F.cross_entropy(logits.view(-1, 8192), targets.view(-1))
loss.backward()
```

## Configuration

```python
cfg = GlobularConfig(
    dim=256,
    num_agents=128,
    agent_dim=256,
    steps=6,
    num_heads=8,
    use_evolution=True,
    use_novelty_search=True,
    use_gqa=True,
)
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dim` | Model dimension | Required |
| `num_agents` | Number of agents | 128 |
| `agent_dim` | Agent embedding dim | 256 |
| `steps` | Evolution steps | 6 |
| `num_heads` | Attention heads | 8 |
| `use_evolution` | Enable GA | True |
| `use_novelty_search` | Enable novelty | True |
| `use_gqa` | Grouped query attention | True |

## License

MIT