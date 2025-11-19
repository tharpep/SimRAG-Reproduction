# SimRAG Reproduction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

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
- **Two-Stage Fine-Tuning**: Instruction following (Stage 1) and domain adaptation (Stage 2)
- **Synthetic QA Generation**: Self-improving data generation from domain documents
- **Multiple AI Providers**: Unified interface for Ollama (local), Purdue GenAI, and HuggingFace
- **Hardware-Aware**: Optimized for both GPU (RTX 3080) and CPU setups

## Installation

### Prerequisites

- Python 3.9+
- [Poetry](https://python-poetry.org/docs/#installation) (recommended) or pip
- [Ollama](https://ollama.ai/) (optional, for local inference):
  ```bash
  ollama pull llama3.2:1b  # For CPU/laptop
  ollama pull qwen2.5:7b   # For GPU systems
  ```

### Setup

```bash
# Clone repository
git clone <repository-url>
cd SimRAG-Reproduction

# Install with Poetry
poetry install
poetry shell

# Or with pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
# Hardware configuration
USE_LAPTOP=true  # true for laptop (llama3.2:1b), false for PC (qwen2.5:7b)

# AI Provider
USE_OLLAMA=true  # true for Ollama (local), false for Purdue API
PURDUE_API_KEY=your-api-key-here  # Required if USE_OLLAMA=false

# Vector store
USE_PERSISTENT=true
COLLECTION_NAME=simrag_docs

# Fine-tuning (optional)
TUNING_BATCH_SIZE=4
TUNING_EPOCHS=3
TUNING_DEVICE=auto  # auto, cpu, cuda, mps
```

**Note**: Never commit `.env` files or API keys.

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

## Troubleshooting

**No providers available**: Set `USE_OLLAMA=true` in `.env` or provide `PURDUE_API_KEY`

**Model not found (Ollama)**: Run `ollama pull llama3.2:1b` or `ollama pull qwen2.5:7b`

**Out of memory**: Reduce `TUNING_BATCH_SIZE=1` or set `TUNING_DEVICE=cpu` in `.env`

**Qdrant errors**: Delete `data/qdrant_db/` or set `USE_PERSISTENT=false`

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

MIT License - Copyright © 2025 Pryce Tharpe

---

**Note**: This is an educational reproduction. Results may differ from the original paper due to hardware constraints and implementation simplifications.
