# SimRAG Reproduction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

A simplified, educational implementation of **SimRAG: Self-improving Retrieval-Augmented Generation** on consumer hardware. This project reproduces the core methodology from the [SimRAG paper](https://arxiv.org/abs/2410.17952) (Xu et al., NAACL 2025), testing whether two-stage fine-tuning improves RAG performance when scaled down to smaller models (1.5B parameters).

## About

This implementation is based on:

> **SimRAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains**
> Ran Xu, Hui Liu, Sreyashi Nag, Zhenwei Dai, Yaochen Xie, Xianfeng Tang, Chen Luo, Yang Li, Joyce C. Ho, Carl Yang, Qi He
> *Proceedings of NAACL 2025*
> [arXiv:2410.17952](https://arxiv.org/abs/2410.17952)

**Core Methodology**: Two-stage fine-tuning of RAG systems using (1) general instruction-following and (2) domain adaptation with synthetically generated QA pairs.

**This Reproduction**: Tests the SimRAG methodology on consumer hardware (RTX 3080, 10GB VRAM) using QLoRA-optimized Qwen 2.5 1.5B-Instruct. All code is original work written from scratch based on the paper's methodology.

## Features

- **RAG System**: Document ingestion, semantic search (Qdrant), and context-aware generation
- **QLoRA Fine-Tuning**: Memory-efficient training with 4-bit quantization and LoRA adapters
- **Two-Stage Training**: Instruction following (Stage 1) + domain adaptation (Stage 2)
- **Synthetic QA Generation**: Automatic QA pair generation from domain documents
- **Self-Improvement Loop**: Optional multi-round Stage 2 training (configurable via `SIMRAG_IMPROVEMENT_ROUNDS`)
- **Hardware-Efficient**: Runs on consumer GPUs (10GB VRAM) using 4-bit quantization


## Installation

### 🐳 Recommended: Docker Installation (Best for Reproducibility)

**Docker is the easiest and most reproducible way to get started.** It handles all dependencies automatically and ensures consistent results across different machines.

**Prerequisites**:
- Docker Engine 20.10+ and Docker Compose 2.0+
- (Optional) NVIDIA Docker runtime for GPU acceleration: [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

**Quick Start with Docker**:

```bash
# 1. Clone repository
git clone <repository-url>
cd SimRAG-Reproduction

# 2. (Optional) Create .env file for API keys
cp .env.example .env
# Edit .env and add your API keys if desired (optional - HuggingFace works without)

# 3. Build the Docker image
docker-compose build simrag

# 4. Run commands (works on both GPU and CPU - automatically detects)
docker-compose run --rm simrag poetry run simrag config
```

**Run Commands in Docker**:

```bash
# View configuration
docker-compose run --rm simrag poetry run simrag config

# Run the recommended workflow
docker-compose run --rm simrag poetry run simrag experiment stage1
docker-compose run --rm simrag poetry run simrag experiment stage2
docker-compose run --rm simrag poetry run simrag experiment test

# Or use interactive shell
docker-compose run --rm simrag bash
# Then run: poetry run simrag experiment stage1, etc.
```

**Note**: The Docker image works on both GPU and CPU systems. If you have an NVIDIA GPU with Docker GPU support, it will automatically use the GPU. Otherwise, it will fall back to CPU (slower but works fine). All data (models, results, logs) is stored in mounted volumes (`./data`, `./tuned_models`, `./logs`, `./comparison_results`) and persists between container runs.

### Alternative: Poetry Installation (For Development)

If you prefer to install directly on your system:

**Prerequisites**:
- **Python 3.12** (required for PyTorch CUDA support): Download from [python.org](https://www.python.org/downloads/)
- **Poetry** (dependency management): `pip install poetry`

**Setup**:

```bash
# 1. Clone repository
git clone <repository-url>
cd SimRAG-Reproduction

# 2. Install dependencies
poetry install

# 3. Install shell plugin (required for 'poetry shell' command)
poetry self add poetry-plugin-shell

# 4. Activate environment
poetry shell

# 5. Verify installation
simrag --help
```

**Note**: Poetry handles PyTorch CUDA installation automatically. The `poetry install` step may take 5-10 minutes on first run.

**Optional: AI Provider for QA Generation**:
- **HuggingFace**: Default (no setup required, works offline)
- **Claude API**: Get key from https://console.anthropic.com/ → Add `CLAUDE_API_KEY` to `.env`
- **Purdue GenAI API**: Add `PURDUE_API_KEY` to `.env` (free for Purdue users)
- Set `QA_PROVIDER=claude` or `purdue` in `.env` to override default

## Configuration

**Note**: A `.env` file is **optional**. The project works with sensible defaults (Qwen 2.5 1.5B, HuggingFace provider, in-memory vector store). Create `.env` only to customize settings.

**Common Configuration** (copy from `.env.example`):

```bash
# Model size (default: small)
MODEL_SIZE=small  # "small" = Qwen 2.5 1.5B, "medium" = Qwen 2.5 7B

# AI Provider for QA generation (default: huggingface)
QA_PROVIDER=huggingface  # Options: "claude", "purdue", "huggingface"
CLAUDE_API_KEY=sk-...   # Optional - for Claude API
PURDUE_API_KEY=...      # Optional - for Purdue GenAI API

# Training settings (defaults are optimized)
TUNING_BATCH_SIZE=4     # Reduce to 1-2 if OOM errors
TUNING_EPOCHS=3
LORA_R=16               # LoRA rank (8-64)
LORA_ALPHA=32           # LoRA scaling

# Self-improvement (default: 1 = no self-improvement)
SIMRAG_IMPROVEMENT_ROUNDS=1  # Set to 2+ for iterative refinement

# Reproducibility
RANDOM_SEED=42
```

**Important Notes**:
- **Model Testing**: This reproduction was validated using **Qwen 2.5 1.5B only**. The 7B configuration is provided but not experimentally validated.
- **Default Model**: Qwen 2.5 1.5B-Instruct requires no authentication and works out-of-the-box.
- **Never commit** `.env` files or API keys to version control.

## Quick Start

After installation (see [Installation](#installation) above), run the recommended workflow:

```bash
# Using Docker (recommended for new users)
docker-compose run --rm simrag poetry run simrag experiment stage1
docker-compose run --rm simrag poetry run simrag experiment stage2
docker-compose run --rm simrag poetry run simrag experiment test

# Using Poetry
poetry shell  # Activate environment first
simrag experiment stage1
simrag experiment stage2
simrag experiment test
```

**What this does:**
- **Stage 1** (~3-4 hours): Fine-tunes base model on general instruction-following
- **Stage 2** (~3-4 hours): Fine-tunes Stage 1 model on domain documents  
- **Test** (~5-15 min): Evaluates and compares models (automatically displays results)

**Note**: On first run, models will be downloaded from HuggingFace Hub (may take a few minutes).

## Usage

### CLI Commands

```bash
simrag test                    # Interactive test selection
simrag test --all              # Run all tests
simrag config                  # View configuration
simrag experiment run          # Train Stage 1 → Stage 2 (same as 'simrag')
simrag experiment stage1       # Stage 1 training only
simrag experiment stage2       # Stage 2 training only
simrag experiment test         # Full test: baseline → fine-tuned → comparison (auto-displays results)
simrag experiment baseline     # Baseline RAG test only
simrag experiment compare      # Compare existing results (if you have JSON files)
simrag experiment export       # Export model for Colab
simrag experiment results      # View previously saved comparison results
```

### Programmatic Usage

```python
from simrag_reproduction.rag.rag_setup import BasicRAG

# Initialize and use RAG system
rag = BasicRAG()
rag.add_documents(["doc1.txt", "doc2.md"])
response = rag.query("What is the main topic?")
print(response)
```

## Project Structure

```
simrag_reproduction/
├── ai_providers/      # LLM clients (Claude, Purdue API, HuggingFace)
├── rag/              # RAG system (ingestion, retrieval, generation)
├── simrag/           # SimRAG pipeline (Stage 1 & 2 fine-tuning)
├── tuning/            # Model fine-tuning utilities
├── experiments/       # Experiment orchestration
│   ├── baseline/     # Baseline RAG experiments (HuggingFace-based)
│   ├── simrag/       # SimRAG training experiments
│   ├── comparison/   # Result comparison utilities
│   ├── local_testing/ # Local HuggingFace testing
│   └── model_utils.py # Model discovery and export utilities
└── cli/               # Command-line interface
```

## Experiments

### Training Commands

```bash
# Train both stages automatically (~6-8 hours total)
simrag experiment run

# Or train stages individually
simrag experiment stage1       # Stage 1: Instruction following (~3-4 hours)
simrag experiment stage2       # Stage 2: Domain adaptation (~3-4 hours)
```

**Note**: `stage2` will prompt you to select a Stage 1 model if not specified.

### Testing & Evaluation

```bash
# Full evaluation: baseline → fine-tuned → comparison (~5-15 min)
simrag experiment test

# Individual commands
simrag experiment baseline     # Baseline RAG only (~2-3 min)
simrag experiment compare      # Compare existing results
simrag experiment results      # View saved comparison results
```

### Recommended Workflow

```bash
# 1. Train models
simrag experiment stage1
simrag experiment stage2

# 2. Evaluate and compare
simrag experiment test
```

## Development

```bash
# Run tests
pytest

# Code quality
black simrag_reproduction/
ruff check simrag_reproduction/
mypy simrag_reproduction/
```

## Hardware Requirements

### Minimum (CPU-only)
- **RAM**: 16GB
- **Storage**: 10GB
- **Performance**: Slow training (~hours), slow inference

### Recommended (GPU)
- **GPU**: 10GB+ VRAM (RTX 3080, RTX 4070, etc.)
- **RAM**: 16GB
- **Storage**: 10GB
- **Performance**: Fast training (~minutes), fast inference

### Memory Usage (with QLoRA + 4-bit Quantization)
| Model | Training VRAM | Inference VRAM | Adapter Size |
|-------|--------------|----------------|--------------|
| Qwen 2.5 1.5B | ~3-4GB | ~1.5GB | ~100MB |
| Qwen 2.5 7B | ~8-10GB | ~4-5GB | ~400MB |

**Note**: Without QLoRA, memory requirements are 3-5x higher and may not fit on consumer GPUs.

**Note**: This reproduction was validated using the **1.5B model only**. The 7B configuration is provided but not experimentally validated.

### Software Environment

**Python**: 3.12 (required for PyTorch CUDA support)

**CUDA**: 12.1+ (if using GPU)
- PyTorch is configured to use CUDA 12.1 via Poetry
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Expected Runtimes** (RTX 3080, 10GB VRAM, Qwen 2.5 1.5B):
- Stage 1: ~3-4 hours | Stage 2: ~3-4 hours | Testing: ~5-15 minutes
- 7B model: ~2-3x longer

## Troubleshooting

### Setup Issues

**"command not found: simrag"**:
- Make sure you've run `poetry shell` to activate the virtual environment
- Verify installation: `poetry install` should complete without errors
- Check Poetry environment: `poetry env info`

**"poetry shell" doesn't work**:
- Install Poetry shell plugin: `poetry self add poetry-plugin-shell` (required - see Installation section)
- Or use `poetry run simrag` instead of `simrag` (no shell needed)

**"poetry env use python3.12" fails**:
- Ensure Python 3.12 is installed: `python3.12 --version`
- On Windows, try: `poetry env use C:\Python312\python.exe` (adjust path)
- Verify Python is in PATH: `where python3.12` (Windows) or `which python3.12` (Linux/Mac)

**Poetry installation issues**:
- Try: `pip install --upgrade poetry`
- Or use official installer: `curl -sSL https://install.python-poetry.org | python3 -`

### Runtime Issues

**CUDA Out of Memory**:
- Reduce batch size: `TUNING_BATCH_SIZE=1` in `.env`
- Use smaller model: `MODEL_SIZE=small` (default)
- Force CPU mode: `TUNING_DEVICE=cpu` (very slow)

**Docker issues**:
- **"nvidia-docker not found"**: GPU is optional. Container works on CPU (slower). For GPU: [Install NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **"Permission denied"**: Fix volume permissions: `sudo chown -R $USER:$USER ./data ./tuned_models ./logs`
- **"Container exits"**: Use interactive shell: `docker-compose run --rm simrag bash`

**Provider issues**:
- System uses HuggingFace by default (no setup required)
- For Claude: Install `pip install anthropic` and set `CLAUDE_API_KEY` in `.env`

**Vector store errors**:
- Default is in-memory (no `data/qdrant_db/` folder created)
- If persistent mode fails: Delete `data/qdrant_db/` or set `USE_PERSISTENT=false`

## Model Management

Trained models are stored as LoRA adapters in `tuned_models/`:

```
tuned_models/
├── model_1b/
│   ├── stage_1/v1.0/, v1.1/, ...  (~100MB each)
│   └── stage_2/v1.0/, v1.1/, ...  (~100MB each)
└── model_8b/                       (~400MB each)
```

- `model_registry.json` tracks versions, parameters, and metrics
- **Local Testing**: `simrag experiment test` (HuggingFace + 4-bit quantization)
- **Export for Colab**: `simrag experiment export` (creates ZIP files)

## Logging

Logs are automatically created in `logs/rag/` and `logs/tuning/`. They include query details, training metrics, and performance data. Logs rotate at 1MB (keeps 3 backups).

## Code Attribution

### Original Implementation

**All code was written from scratch for this reproduction study.** The SimRAG paper does not provide a public implementation, so all components were developed independently based on the paper's methodology.

**Key Components** (all original):
- `simrag_reproduction/simrag/` - Two-stage training pipeline (Stage 1 & 2)
- `simrag_reproduction/rag/` - RAG system (ingestion, retrieval, generation)
- `simrag_reproduction/tuning/` - QLoRA fine-tuning and model registry
- `simrag_reproduction/experiments/` - Experiment orchestration and comparison
- `simrag_reproduction/ai_providers/` - Multi-provider LLM interface
- `simrag_reproduction/cli/` - Command-line interface

### Third-Party Libraries

Standard libraries used as dependencies (unmodified):
- PyTorch, HuggingFace Transformers, PEFT, bitsandbytes
- Qdrant, ChromaDB, sentence-transformers
- Poetry, Typer, datasets

### AI-Assisted Development

Claude Sonnet 4.5 (via Cursor IDE) was used for code generation, debugging, and documentation. All architectural decisions are original.

## Citation

If you use this code, please cite the original SimRAG paper:

```bibtex
@article{xu2024simrag,
  title={SimRAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains},
  author={Xu, Ran and Liu, Hui and Nag, Sreyashi and Dai, Zhenwei and Xie, Yaochen and Tang, Xianfeng and Luo, Chen and Li, Yang and Ho, Joyce C. and Yang, Carl and He, Qi},
  journal={Proceedings of NAACL 2025},
  year={2025},
  url={https://arxiv.org/abs/2410.17952}
}
```

## Creating Submission ZIP

To create a ZIP file of the codebase for submission (excludes `.gitignore` patterns, `project_docs/`, and `.github/`):

```bash
python create_submission_zip.py
```

Or specify a custom output name:

```bash
python create_submission_zip.py SimRAG-Submission.zip
```

The script will:
- Exclude all files/folders listed in `.gitignore`
- Exclude `project_docs/` folder
- Exclude `.github/` folder
- Create a timestamped ZIP file in the project root

## License

MIT License - Copyright © 2025 SimRAG Reproduction Project

---

**Note**: This is an educational reproduction. Results may differ from the original paper due to hardware constraints and implementation simplifications.
