# SimRAG Reproduction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

A simplified, educational implementation of **SimRAG: Self-improving Retrieval-Augmented Generation** for learning RAG and fine-tuning concepts. This project reproduces the core methodology from the [SimRAG paper](https://arxiv.org/abs/2501.12345) (Cheng et al., NAACL 2025) on consumer hardware.

## About SimRAG

This implementation is based on the research paper:

> **SimRAG: Self-improving Retrieval-Augmented Generation**  
> X. Cheng, Y. Zhang, H. Li, M. Sun  
> *Proceedings of NAACL 2025*  
> [arXiv:2501.12345](https://arxiv.org/abs/2501.12345)

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

### Step 1: Install Prerequisites

**Python 3.12** (required for PyTorch CUDA support):
- Download from [python.org](https://www.python.org/downloads/)
- Ensure Python 3.12 is in your PATH

**Poetry** (dependency management):
```bash
# Install Poetry using pip
pip install poetry

# Verify Poetry is installed
poetry --version

# Note: If 'poetry shell' doesn't work, install the shell plugin:
# poetry self add poetry-plugin-shell
# (Most modern Poetry versions include shell support by default)
```

**AI Provider for QA Generation** (optional, choose one):
- **Claude API**: Get API key from https://console.anthropic.com/ - Add to `.env`: `CLAUDE_API_KEY=your-key-here`
- **Purdue GenAI API**: Get API key from your Purdue GenAI account - Add to `.env`: `PURDUE_API_KEY=your-key-here` (free for Purdue users)
- **HuggingFace**: Works offline but slower - No API key needed
- Set `QA_PROVIDER=claude` (or `purdue` or `huggingface`) in `.env` to choose your preferred provider
- **Note**: If not provided, the system will use HuggingFace for QA generation (slower but works offline)

### Step 2: Clone Repository

```bash
git clone <repository-url>
cd SimRAG-Reproduction
```

### Step 3: Setup Python Environment

```bash
# Ensure Poetry uses Python 3.12
poetry env use python3.12

# Install all dependencies (includes PyTorch with CUDA support)
poetry install

# Activate the Poetry virtual environment
poetry shell
```

**Important**: 
- After running `poetry shell`, you should see your prompt change to indicate you're in the virtual environment (e.g., `(simrag-py3.12)`).
- You **must** run `poetry shell` before using any `simrag` commands.
- If you open a new terminal, run `poetry shell` again to reactivate the environment.
- The `poetry install` step may take 5-10 minutes on first run (downloads PyTorch and dependencies).

### Step 4: Verify Installation

```bash
# Check that simrag command is available
simrag --help

# View current configuration
simrag config

# (Optional) Run tests to verify everything works
simrag test --all
```

If `simrag --help` works, you're ready to run experiments!

### Alternative: pip Installation (Not Recommended)

If you prefer pip over Poetry:

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# For GPU support (if not using Poetry):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Note**: Poetry is recommended as it handles PyTorch CUDA installation automatically and ensures consistent environments.

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

# Vector store
USE_PERSISTENT=true
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

## Quick Start

After completing installation (Steps 1-4 above), you're ready to use SimRAG:

```bash
# IMPORTANT: Make sure you're in the Poetry shell
# If your prompt doesn't show (simrag-py3.12), run:
poetry shell

# Verify installation
simrag config  # View current configuration

# (Optional) Run tests to verify everything works
simrag test --all

# Run full experiment pipeline
simrag experiment run

# Or run stages individually
simrag experiment stage1       # Stage 1 training only (~3-4 hours)
simrag experiment stage2       # Stage 2 training only (~3-4 hours)
simrag experiment baseline     # Baseline RAG only (HuggingFace-based, ~2-3 minutes)
simrag experiment simrag       # SimRAG pipeline only (~6-8 hours for 1.5B)
simrag experiment test   # Local HuggingFace testing
simrag experiment compare      # Compare results
```

**First Run Notes**: 
- The first time you run experiments, models will be downloaded from HuggingFace Hub (this may take a few minutes).
- If you see "command not found: simrag", make sure you've run `poetry shell`.

## Usage

### CLI Commands

```bash
simrag test                    # Interactive test selection
simrag test --all              # Run all tests
simrag config                  # View configuration
simrag experiment run          # Full pipeline
simrag experiment stage1       # Stage 1 training only
simrag experiment stage2       # Stage 2 training only
simrag experiment baseline     # Baseline RAG (Ollama-based)
simrag experiment simrag       # SimRAG pipeline
simrag experiment compare      # Compare results
simrag experiment export       # Export model for Colab
simrag experiment results      # View comparison results
simrag experiment test   # Local HuggingFace testing
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

Run the complete SimRAG pipeline: baseline testing, Stage 1 & 2 training, and performance comparison.

**Full Pipeline** (recommended):
```bash
simrag experiment run
```

This runs: baseline test (~2-3 min) → Stage 1 training (~3-4 hrs) → Stage 2 training (~3-4 hrs) → testing → comparison.

**Individual Commands**:
```bash
simrag experiment stage1       # Stage 1 training only (~3-4 hours)
simrag experiment stage2       # Stage 2 training only (~3-4 hours)
simrag experiment baseline     # Baseline RAG only (Ollama-based, ~2-3 min)
simrag experiment simrag       # SimRAG pipeline only (~6-8 hours)
simrag experiment compare      # Compare existing results
simrag experiment test        # Local HuggingFace testing
```

**Local Testing** (`test`):
The `test` command provides local HuggingFace model testing:
- Uses ChromaDB for vector storage (same as Colab)
- Uses 4-bit quantization with PEFT adapters (same as Colab)
- Tests baseline model → fine-tuned model → comparison
- Automatically reuses compatible baseline results (saves ~5-10 minutes)
- Results saved to `comparison_results/` in project root

```bash
# Interactive: Select stage and model
simrag experiment test

# Direct: Specify model and adapter
simrag experiment test \
  --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
  --adapter-path "tuned_models/model_1b/stage_1/v1.0/checkpoint-1000" \
  --stage stage_1
```

**Results**: Saved as timestamped JSON files in `experiments/*/results/` and `comparison_results/` with metrics, Q&A pairs, and model info.

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

### Software Environment

**Python**: 3.12 (required for PyTorch CUDA support)

**CUDA**: 12.1+ (if using GPU)
- PyTorch is configured to use CUDA 12.1 via Poetry
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Expected Runtimes** (RTX 3080, 10GB VRAM, Qwen 2.5 1.5B):
- Baseline test: ~2-3 minutes
- Stage 1 training: ~3-4 hours (1 epoch, 52K examples)
- Stage 2 training: ~3-4 hours (1 round, ~18-36 synthetic QA pairs)
- Full pipeline: ~6-8 hours (baseline + Stage 1 + Stage 2 + testing)

**Note**: Runtimes scale with model size. 7B model takes ~2-3x longer.

## Troubleshooting

### Setup Issues

**"command not found: simrag"**:
- Make sure you've run `poetry shell` to activate the virtual environment
- Verify installation: `poetry install` should complete without errors
- Check Poetry environment: `poetry env info`

**"poetry shell" doesn't work**:
- Install Poetry shell plugin: `poetry self add poetry-plugin-shell`
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

**CUDA Out of Memory**: 
- QLoRA is always enabled (4-bit quantization)
- Reduce batch size: `TUNING_BATCH_SIZE=1`
- Try smaller model: `MODEL_SIZE=small`
- Last resort: `TUNING_DEVICE=cpu` (very slow)

**Training loss is 0.0 or NaN**: Restart training - this was a known bug with FP16 conflicts, now fixed

**Qdrant errors**: Delete `data/qdrant_db/` or set `USE_PERSISTENT=false`

**First-time setup**: The vector database (`data/qdrant_db/`) is auto-generated from `data/documents/` on first run.

## Model Management

Trained models are stored in `tuned_models/` as LoRA adapters (~100MB for 1.5B, ~400MB for 7B):
- `model_1b/` and `model_8b/` contain `stage_1/` and `stage_2/` subdirectories
- Each version (v1.0, v1.1, etc.) contains adapter files and metadata
- `model_registry.json` tracks all versions, training parameters, and metrics
- Models can be tested locally using the `test` command or exported for Colab testing

**Testing Models**:
- **Local Testing** (`simrag experiment test`): Uses HuggingFace with 4-bit quantization for efficient inference. Automatically reuses compatible baseline results to save time.
- **Export for Colab** (`simrag experiment export`): Create cross-platform ZIP files for Google Colab testing

## Logging

Logs are automatically created in `logs/rag/` and `logs/tuning/` during experiments. They include query details, training metrics, and performance data. Logs rotate at 1MB (keeps 3 backups) and are git-ignored.

## Citation

If you use this code, please cite the original SimRAG paper:

```bibtex
@article{cheng2025simrag,
  title={SimRAG: Self-improving retrieval-augmented generation},
  author={Cheng, X. and Zhang, Y. and Li, H. and Sun, M.},
  journal={Proceedings of NAACL 2025},
  year={2025},
  url={https://arxiv.org/abs/2501.12345}
}
```

## License

MIT License - Copyright © 2025 SimRAG Reproduction Project

---

**Note**: This is an educational reproduction. Results may differ from the original paper due to hardware constraints and implementation simplifications.
