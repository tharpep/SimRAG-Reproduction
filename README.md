# SimRAG Reproduction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

A simplified, educational implementation of **SimRAG: Self-improving Retrieval-Augmented Generation** for learning RAG and fine-tuning concepts. This project reproduces the core methodology from the [SimRAG paper](https://arxiv.org/abs/2410.17952) (Xu et al., NAACL 2025) on consumer hardware.

## About SimRAG

This implementation is based on the research paper:

> **SimRAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains**  
> Ran Xu, Hui Liu, Sreyashi Nag, Zhenwei Dai, Yaochen Xie, Xianfeng Tang, Chen Luo, Yang Li, Joyce C. Ho, Carl Yang, Qi He  
> *Proceedings of NAACL 2025*  
> [arXiv:2410.17952](https://arxiv.org/abs/2410.17952)

SimRAG introduces a self-improving framework that fine-tunes RAG systems through two stages: (1) general instruction-following, and (2) domain adaptation using synthetically generated QA pairs. This reproduction verifies the core hypothesis that two-stage fine-tuning improves RAG performance on domain-specific documents compared to vanilla RAG.

## Features

- **RAG System**: Document ingestion, semantic search with Qdrant, and context-aware generation
- **QLoRA Fine-Tuning**: Memory-efficient training using 4-bit quantization and LoRA adapters
- **Two-Stage Training**: Instruction following (Stage 1) and domain adaptation with integrated self-improvement (Stage 2)
- **Self-Improvement Loop**: Stage 2 can run multiple rounds, each using the improved model to generate better synthetic QA
- **Multiple AI Providers**: Unified interface for Claude, Purdue GenAI API, and HuggingFace
- **Synthetic QA Generation**: Self-improving data generation from domain documents
- **Hardware-Efficient**: Runs on consumer GPUs (10GB VRAM) with 4-bit quantization


## Installation

### 🐳 Recommended: Docker Installation (Best for Reproducibility)

**Docker is the easiest and most reproducible way to get started.** It handles all dependencies automatically and ensures consistent results across different machines.

**Prerequisites**:
- Docker Engine 20.10+ and Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support): [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

**Quick Start with Docker**:

```bash
# 1. Clone repository
git clone <repository-url>
cd SimRAG-Reproduction

# 2. (Optional) Create .env file for API keys
cp .env.example .env
# Edit .env and add your API keys if desired (optional - HuggingFace works without)

# 3. Build and run (CPU version - works on any machine)
docker-compose up --build simrag-cpu

# Or for GPU support (if you have NVIDIA GPU):
docker-compose up --build simrag
```

**Run Commands in Docker**:

```bash
# View configuration
docker-compose run --rm simrag-cpu poetry run simrag config

# Run the recommended workflow
docker-compose run --rm simrag-cpu poetry run simrag experiment stage1
docker-compose run --rm simrag-cpu poetry run simrag experiment stage2
docker-compose run --rm simrag-cpu poetry run simrag experiment test

# Or use interactive shell
docker-compose run --rm simrag-cpu bash
# Then run: simrag experiment stage1, etc.
```

**Docker Benefits**:
- ✅ No Python/Poetry setup required
- ✅ Consistent environment across machines
- ✅ Automatic dependency management
- ✅ GPU support included (with nvidia-docker2)
- ✅ Perfect for reproducibility

**Data Persistence**: All data (models, results, logs) is stored in mounted volumes (`./data`, `./tuned_models`, `./logs`, `./comparison_results`) and persists between container runs.

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

**AI Provider for QA Generation** (optional, choose one):
- **Claude API**: Get API key from https://console.anthropic.com/ - Add to `.env`: `CLAUDE_API_KEY=your-key-here`
- **Purdue GenAI API**: Get API key from your Purdue GenAI account - Add to `.env`: `PURDUE_API_KEY=your-key-here` (free for Purdue users)
- **HuggingFace**: Works offline but slower - No API key needed
- Set `QA_PROVIDER=claude` (or `purdue` or `huggingface`) in `.env` to choose your preferred provider
- **Note**: If not provided, the system will use HuggingFace for QA generation (slower but works offline)

### Alternative: pip Installation (Not Recommended)


## Configuration

**Note**: A `.env` file is **optional** for basic usage. The project works with sensible defaults. Create a `.env` file only if you need to customize settings.

Create a `.env` file in the project root (optional):

```bash
# Model size configuration
MODEL_SIZE=small  # "small" for Qwen/Qwen2.5-1.5B-Instruct, "medium" for Qwen/Qwen2.5-7B-Instruct (both non-gated)

# AI Provider (for synthetic QA generation during training)
QA_PROVIDER=purdue  # Options: "claude", "purdue", "huggingface" (default: "purdue")
CLAUDE_API_KEY=your-claude-api-key-here  # Optional - for Claude API (get from https://console.anthropic.com/)
PURDUE_API_KEY=your-purdue-api-key-here  # Optional - for Purdue GenAI API (free for Purdue users)

# HuggingFace (optional - only needed if using Llama models)
HF_TOKEN=your-huggingface-token-here  # Optional: Only needed for gated Llama models. Get from https://huggingface.co/settings/tokens

# Vector store (optional - defaults to in-memory)
USE_PERSISTENT=false  # Set to true for persistent Qdrant storage (creates data/qdrant_db/ folder)
COLLECTION_NAME=simrag_docs


# Local testing settings (optional)
REUSE_BASELINE=true  # Reuse compatible baseline if available (saves ~5-10 min, default: true)
BASELINE_MAX_AGE_DAYS=7  # Maximum age in days for reusable baseline (default: 7)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Embedding model for vector store
LOCAL_TESTING_MAX_TOKENS=512  # Max tokens for local testing (default: 512)

# Fine-tuning (optional)
TUNING_BATCH_SIZE=4
TUNING_EPOCHS=3
TUNING_DEVICE=auto  # auto, cpu, cuda, mps

# QLoRA settings (optional - defaults are optimized)
# Note: QLoRA is always enabled (4-bit + LoRA adapters)
LORA_R=16  # LoRA rank (8-64, higher = more expressive)
LORA_ALPHA=32  # LoRA scaling (typically 2x lora_r)

# Self-improvement settings (optional)
SIMRAG_IMPROVEMENT_ROUNDS=1  # Number of Stage 2 rounds (1=no self-improvement, 2+=iterative refinement)

# Reproducibility (optional)
RANDOM_SEED=42  # Random seed for reproducible results (default: 42)
# Note: cuDNN deterministic mode is NOT used (causes 10x slowdown). Random seeds provide sufficient reproducibility.
```

**Note**: Never commit `.env` files or API keys.

**Model Selection**:
- **Default (Qwen 2.5 Instruct)**: No authentication needed - works out of the box! Uses standard generation (non-thinking mode) ✅
- **Llama models**: If you prefer Llama, set `HF_TOKEN` in `.env` and accept license at https://huggingface.co/meta-llama/Llama-3.2-1B

**Note on Model Testing**: This reproduction was tested and validated using the **1.5B model** (`Qwen/Qwen2.5-1.5B-Instruct`). While the codebase supports 7B models and can be adapted for other models, only the 1.5B model was used for experimental validation and conclusions in this reproduction study.

## Quick Start

After installation (see [Installation](#installation) above), run the recommended workflow:

```bash
# Using Docker (recommended for new users)
docker-compose run --rm simrag-cpu poetry run simrag experiment stage1
docker-compose run --rm simrag-cpu poetry run simrag experiment stage2
docker-compose run --rm simrag-cpu poetry run simrag experiment test

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

**First Run**: Models will be downloaded from HuggingFace Hub (may take a few minutes).

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

**Important**: The commands are modular - you can run stages separately or together.

### Training Commands

**`experiment run`** or **`experiment simrag`** (same thing):
- Runs Stage 1 training (instruction following) → Stage 2 training (domain adaptation)
- **Does NOT run**: baseline testing, local testing, or comparison
- Time: ~6-8 hours for 1.5B model
- Use this to train both stages automatically

**Individual Training Stages**:
```bash
simrag experiment stage1       # Stage 1 training only (~3-4 hours)
simrag experiment stage2       # Stage 2 training only (~3-4 hours)
```

**Note**: `experiment stage2` will prompt you to select a Stage 1 model if you haven't specified one.

### Testing & Evaluation Commands

**`experiment test`** (recommended for evaluation):
- Runs full test flow: baseline → fine-tuned model → comparison
- **Automatically displays comparison results** (no need to run `experiment results` separately)
- Saves results to `comparison_results/` directory
- Time: ~5-15 minutes (reuses baseline if compatible)
- Use this after training to evaluate your models

**Other Testing Commands**:
```bash
simrag experiment baseline     # Baseline RAG test only (~2-3 min)
simrag experiment compare      # Compare existing results (if you have baseline/simrag JSON files)
simrag experiment results      # View previously saved comparison results
```

### Typical Workflow

**Recommended: Stage-by-Stage** (see [Quick Start](#quick-start) above)
```bash
simrag experiment stage1
simrag experiment stage2
simrag experiment test
```

**Alternative: Automated Training**
```bash
simrag experiment run  # Trains both stages automatically
simrag experiment test
```

**Advanced Testing Options**:
```bash
# Interactive: Select stage and model
simrag experiment test

# Direct: Specify model and adapter
simrag experiment test \
  --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
  --adapter-path "tuned_models/model_1b/stage_1/v1.0/checkpoint-1000" \
  --stage stage_1
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

**Testing Note**: This reproduction was validated using the **1.5B model only**. The 7B model configuration is provided for users who want to experiment, but experimental results and conclusions are based solely on 1.5B model testing.

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

**No providers available**: The system uses HuggingFace by default. For faster QA generation during training, set `CLAUDE_API_KEY` or `PURDUE_API_KEY` in `.env` and set `QA_PROVIDER=claude` or `QA_PROVIDER=purdue`. The `test` command always uses HuggingFace directly.

**Claude API not working**: If you want to use Claude, install the anthropic package: `pip install anthropic` or `poetry add anthropic`. The package is marked as optional in pyproject.toml.

**Docker issues**:
- **"nvidia-docker not found"**: Install NVIDIA Docker runtime. For GPU support, you need `nvidia-docker2` or Docker with `--gpus all` support.
- **"CUDA out of memory"**: Reduce batch size in `.env`: `TUNING_BATCH_SIZE=1` or use CPU version: `docker-compose up simrag-cpu`
- **"Permission denied"**: Ensure Docker has access to mounted volumes. On Linux, you may need to adjust permissions: `sudo chown -R $USER:$USER ./data ./tuned_models ./logs`
- **"Container exits immediately"**: Use `docker-compose run --rm simrag bash` for interactive shell, or specify a command: `docker-compose run --rm simrag poetry run simrag --help`

**CUDA Out of Memory**: 
- QLoRA is always enabled (4-bit quantization)
- Reduce batch size: `TUNING_BATCH_SIZE=1`
- Try smaller model: `MODEL_SIZE=small`
- Last resort: `TUNING_DEVICE=cpu` (very slow)

**Training loss is 0.0 or NaN**: Restart training - this was a known bug with FP16 conflicts, now fixed

**Qdrant errors**: Delete `data/qdrant_db/` if it exists, or set `USE_PERSISTENT=false` (default is in-memory, so folder shouldn't be created)

**Note**: The system uses in-memory Qdrant by default during training. The `data/qdrant_db/` folder is only created if `USE_PERSISTENT=true` is set in `.env`.

## Model Management

Trained models are stored in `tuned_models/` as LoRA adapters (~100MB for 1.5B, ~400MB for 7B):
- `model_1b/` and `model_8b/` contain `stage_1/` and `stage_2/` subdirectories
- Each version (v1.0, v1.1, etc.) contains adapter files and metadata
- `model_registry.json` tracks all versions, training parameters, and metrics
- Models can be tested locally using the `test` command or exported for Colab testing

**Testing Models**:
- **Local Testing** (`simrag experiment test`): Uses HuggingFace with 4-bit quantization. Automatically reuses compatible baseline results.
- **Export for Colab** (`simrag experiment export`): Create ZIP files for Google Colab testing

## Logging

Logs are automatically created in `logs/rag/` and `logs/tuning/` during experiments. They include query details, training metrics, and performance data. Logs rotate at 1MB (keeps 3 backups) and are git-ignored.

## Code Attribution

### Original Code Written for This Project

All code in this repository was written from scratch for this reproduction study. The SimRAG paper does not provide a public implementation, so all code was developed independently based on the paper's methodology.

**Original Implementation Components:**
- **SimRAG Pipeline** (`simrag_reproduction/simrag/`): Two-stage training pipeline including instruction following (Stage 1) and domain adaptation (Stage 2)
- **Synthetic QA Generation** (`simrag_reproduction/simrag/synthetic_qa_generation.py`): Logic for generating question-answer pairs from domain documents with filtering
- **Experiment Orchestration** (`simrag_reproduction/experiments/`): Baseline testing, SimRAG training, and result comparison utilities
- **RAG System** (`simrag_reproduction/rag/`): Document ingestion, vector storage, retrieval, and context-aware generation
- **Model Fine-Tuning** (`simrag_reproduction/tuning/`): QLoRA training utilities, model registry, and adapter management
- **CLI Interface** (`simrag_reproduction/cli/`): Command-line interface for running experiments
- **Configuration Management** (`simrag_reproduction/config.py`): Configuration system with environment variable overrides
- **AI Provider Gateway** (`simrag_reproduction/ai_providers/`): Unified interface for multiple LLM providers (Claude, Purdue API, HuggingFace)

### Third-Party Libraries Used (Not Modified)

These libraries are used as dependencies but were not modified:
- **PyTorch, Transformers (HuggingFace)** - Model training and inference
- **PEFT, bitsandbytes** - QLoRA fine-tuning and 4-bit quantization
- **ChromaDB, Qdrant** - Vector stores for document retrieval
- **sentence-transformers** - Embedding generation
- **Poetry** - Dependency management
- **Typer** - CLI framework
- **datasets** - Dataset loading and processing

### AI-Assisted Development

Claude Sonnet 4.5 (via Cursor IDE) was used as a development assistant for code generation, debugging, and documentation. All architectural decisions and implementations are original.

### Adapted Code

**None** - All implementation is original work for this reproduction study. No code was copied or adapted from other repositories.

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
