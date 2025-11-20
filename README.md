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
- **Two-Stage Training**: Instruction following (Stage 1) and domain adaptation (Stage 2)
- **Ollama Integration**: Automatic conversion of fine-tuned models for fast, reliable inference
- **Synthetic QA Generation**: Self-improving data generation from domain documents
- **Multiple AI Providers**: Unified interface for Ollama (local), Purdue GenAI, and HuggingFace
- **Hardware-Efficient**: Runs on consumer GPUs (10GB VRAM) with 4-bit quantization

## Installation

### Prerequisites

- Python 3.12 (required for PyTorch CUDA support)
- [Poetry](https://python-poetry.org/docs/#installation) (recommended) or pip
- [Ollama](https://ollama.ai/) (optional, for 10x faster baseline testing):
  ```bash
  # Optional: Install Ollama for faster baseline testing (~20s vs ~3min)
  ollama pull qwen2.5:1.5b  # For small model (1.5B)
  ollama pull qwen2.5:7b    # For large model (7B)
  ```
  **Note**: Ollama is NOT required. The baseline will automatically fall back to HuggingFace if Ollama is unavailable.

### Setup

```bash
# Clone repository
git clone <repository-url>
cd SimRAG-Reproduction

# Install with Poetry (uses Python 3.12 automatically)
poetry env use python3.12  # Ensure using Python 3.12
poetry install  # Installs PyTorch with CUDA support automatically
poetry shell

# Or with pip
python3.12 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
# For GPU support: pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Configuration

Create a `.env` file in the project root:

```bash
# Model size configuration
MODEL_SIZE=small  # "small" for Qwen/Qwen2.5-1.5B-Instruct, "medium" for Qwen/Qwen2.5-7B-Instruct (both non-gated)

# AI Provider
USE_OLLAMA=true  # true for Ollama (local), false for Purdue API
PURDUE_API_KEY=your-api-key-here  # Required if USE_OLLAMA=false

# HuggingFace (optional - only needed if using Llama models)
HF_TOKEN=your-huggingface-token-here  # Optional: Only needed for gated Llama models. Get from https://huggingface.co/settings/tokens

# Vector store
USE_PERSISTENT=true
COLLECTION_NAME=simrag_docs

# Baseline provider (optional)
BASELINE_PROVIDER=huggingface  # "huggingface" (default, works everywhere) or "ollama" (10x faster if installed)

# Fine-tuning (optional)
TUNING_BATCH_SIZE=4
TUNING_EPOCHS=3
TUNING_DEVICE=auto  # auto, cpu, cuda, mps

# QLoRA settings (optional - defaults are optimized)
USE_QLORA=true  # Enable QLoRA (4-bit + LoRA adapters) for efficient training
LORA_R=16  # LoRA rank (8-64, higher = more expressive)
LORA_ALPHA=32  # LoRA scaling (typically 2x lora_r)
```

**Note**: Never commit `.env` files or API keys.

**Model Selection**:
- **Default (Qwen 2.5 Instruct)**: No authentication needed - works out of the box! Uses standard generation (non-thinking mode) ✅
- **Llama models**: If you prefer Llama, set `HF_TOKEN` in `.env` and accept license at https://huggingface.co/meta-llama/Llama-3.2-1B

## Quick Start

```bash
# Run tests
simrag test --all

# View configuration
simrag config

# Run full experiment pipeline
simrag experiment run

# Or run stages individually
simrag experiment baseline     # Baseline RAG only
simrag experiment simrag       # SimRAG pipeline only
simrag experiment compare      # Compare results
```

## Usage

### CLI Commands

```bash
simrag test                    # Interactive test selection
simrag test --all              # Run all tests
simrag config                  # View configuration
simrag experiment run          # Full pipeline
simrag experiment baseline     # Baseline RAG
simrag experiment simrag       # SimRAG pipeline
simrag experiment compare      # Compare results
```

### Programmatic Usage

```python
from simrag_reproduction.rag.rag_setup import BasicRAG

# Initialize and use RAG system
rag = BasicRAG()
rag.ingest_documents(["doc1.txt", "doc2.md"])
response = rag.query("What is the main topic?")
print(response)
```

## Project Structure

```
simrag_reproduction/
├── ai_providers/      # LLM clients (Ollama, Purdue, HuggingFace)
├── rag/              # RAG system (ingestion, retrieval, generation)
├── simrag/           # SimRAG pipeline (Stage 1 & 2 fine-tuning)
├── tuning/            # Model fine-tuning utilities
├── experiments/       # Experiment orchestration
└── cli/               # Command-line interface
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

## Troubleshooting

**No providers available**: HuggingFace should be available by default. If needed, set `USE_OLLAMA=true` in `.env` or provide `PURDUE_API_KEY`

**Model not found (Ollama)**: Only if using Ollama - Run `ollama pull qwen2.5:1.5b` (small) or `ollama pull qwen2.5:7b` (medium)

**CUDA Out of Memory**: 
- Ensure QLoRA is enabled: `USE_QLORA=true` (default)
- Reduce batch size: `TUNING_BATCH_SIZE=1`
- Try smaller model: `MODEL_SIZE=small`
- Last resort: `TUNING_DEVICE=cpu` (very slow)

**Training loss is 0.0 or NaN**: Restart training - this was a known bug with FP16 conflicts, now fixed

**Qdrant errors**: Delete `data/qdrant_db/` or set `USE_PERSISTENT=false`

**First-time setup**: The vector database (`data/qdrant_db/`) is auto-generated from `data/documents/` on first run.

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
