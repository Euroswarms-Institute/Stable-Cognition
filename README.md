# 🧠 Globular Reasoning Architecture

<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.0-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3.10+-green?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/huggingface-cli-v0.21+-green?style=flat-square" alt="HF">
</p>

A novel reasoning architecture for language models, inspired by globular clusters in space. Uses populations of latent hypothesis-agents that evolve through attention-like field dynamics.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Agent Field** | Population of latent reasoning agents |
| **Evolution** | Genetic algorithm selection per layer |
| **Auto-Scaling** | No arbitrary caps - scales to 1B+ agents |
| **Hybrid Integration** | Insert into any transformer |
| **Multi-GPU** | Local (RTX 3050) + Cloud (T4/A100) |
| **Rich CLI** | Interactive menu-driven interface |

---

## 🚀 Installation

```bash
# Clone
git clone https://github.com/Euroswarms-Institute/Stable-Cognition.git
cd stable-cognition

# Install
pip install -r requirements.txt

# Or just core deps
pip install torch transformers rich huggingface-hub
```

---

## 📊 Growth Chart

```
Agents → Parameters (per layer at hidden=4096)
═══════════════════════════════════════════════
    16 →   ~3M   ████
    32 →  ~10M   ██████████
    64 →  ~30M   ████████████████████
   128 →  ~80M   ██████████████████████████████████████
   256 → ~180M   ██████████████████████████████████████████████████
   512 → ~450M   ████████████████████████████████████████████████████████████
  1024 → ~1.0B  ███████████████████████████████████████████████████████████████████████████████
  64000 → ~80GB █████████████████████████████████████████████████████████████████████████████████████████... (scales)
═══════════════════════════════════════════════
```

---

## 🎯 Quick Start

### Interactive CLI (Recommended)
```bash
python -m globular.cli
```

### Training
```bash
# Quick training
python -m globular.train --preset lambda --epochs 3

# Custom agents with auto-scaling
python -m globular.train --agents 128 --epochs 1

# Resume from checkpoint
python -m globular.train --resume ./output/globular/checkpoint_e1.pt
```

### Integration
```bash
# Simple mode
python -m globular.cli → Simple Mode

# Or Expert mode
python -m globular.cli → Expert Mode → Configure → Integrate
```

### Generation
```bash
python -m globular.generate ./output/globular-integrated --prompt "The capital of France is"
python -m globular.generate ./output/globular-integrated --chat
```

### Model Hub
```bash
python -m globular.hub --list
python -m globular.hub --download qwen2-0.5b
```

---

## 📖 Usage Examples

### Basic Integration
```python
from globular import (
    GlobularConfig,
    GlobularReasoningBlock,
    apply_globular_to_model,
)

# Config
config = GlobularConfig(
    dim=896,           # Hidden size
    num_agents=32,      # Agent population
    agent_dim=128,       # Agent embedding
    steps=3,            # Evolution steps
)

# Apply to model
model = apply_globular_to_model(llm_model, config)
```

### Auto-Scaling
```python
from globular.cli import auto_scale

# Auto-scale for any agent count
auto = auto_scale(64000)
# → dim=8192, steps=1002, heads=...
```

### Training
```python
from globular.train import TrainingConfig, train_globular

config = TrainingConfig(
    agents=128,
    agent_dim=256,
    epochs=3,
    batch_size=4,
    dataset="synthetic",
)

train_globular(config)
```

---

## 🔧 Configuration Options

| Option | Description | Default |
|--------|-------------|----------|
| `dim` | Hidden dimension | 896 |
| `num_agents` | Agent population | 32 |
| `agent_dim` | Agent embedding | 128 |
| `steps` | Evolution steps | 3 |
| `num_heads` | Agent heads | 8 |
| `insert_layers` | Layer indices | [5, 10] |

### Presets

| Preset | Agents | Dim | Steps | Use Case |
|--------|-------|------|-------|---------|
| Lambda | 32 | 128 | 3 | General reasoning |
| Opencode | 32 | 128 | 3 | Code reasoning |
| Opencode-Reason | 48 | 192 | 4 | Code variant |
| Heavy | 64 | 256 | 4 | High-capacity |
| Heavy+ | Custom | Auto | Auto | Up to 1B agents |

---

## 📁 Project Structure

```
globular/
├── cli.py          # Interactive CLI (Rich)
├── train.py        # Training v2 (checkpoints, metrics)
├── generate.py     # Text generation
├── hub.py         # Model download
├── integration.py # Model integration
│
├── globular/
│   ├── config.py      # Configuration
│   ├── field.py     # Agent field
│   ├── block.py      # Reasoning block
│   ├── model.py      # Example models
│   ├── attention.py  # Custom attention
│   ├── evolution.py # Genetic operators
│   └── novelty.py    # Novelty search
│
└── output/
    └── metrics.jsonl # Training metrics
```

---

## 🏆 Performance

| Model Size | Agents | GPU | Params/Layer | Time/Epoch |
|------------|--------|-----|------|-------------|------------|
| Qwen2-0.5B | 32 | 3050 | ~10M | ~30s |
| Qwen2-1.5B | 64 | T4 | ~30M | ~2min |
| Qwen2-7B | 128 | A100 | ~80M | ~5min |
| Custom | Up to 1B | Multi | Scale | Depends |

---

## 🔗 Links

- [HuggingFace Hub](https://huggingface.co/globular)
- [PyTorch](https://pytorch.org)
- [Transformers](https://huggingface.co/transformers)

---

## 📜 License

Apache 2.0 - See LICENSE file.

---

<p align="center">
  <sub>Built with ⚡ by the Globular Team</sub>
</p>
