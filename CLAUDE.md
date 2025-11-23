# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimRAG Reproduction is an educational implementation of the SimRAG paper (Xu et al., NAACL 2025), demonstrating self-improving Retrieval-Augmented Generation through two-stage fine-tuning of language models. The core innovation is that Stage 2 can run multiple rounds, each using the improved model from the previous round to generate better synthetic QA pairs.

## Development Commands

### Setup
```bash
# Install dependencies (requires Python 3.12)
poetry install              # Recommended (~5-10 min, includes PyTorch CUDA 12.1)
poetry shell               # Activate environment

# Alternative: pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/tests_rag/            # RAG system tests
pytest tests/tests_tuning/         # Fine-tuning tests
pytest tests/tests_simrag/         # SimRAG pipeline tests
pytest tests/tests_integration/    # End-to-end integration tests

# Via CLI
simrag test --all
```

### Running Experiments
```bash
# Full pipeline (baseline → Stage 1 → Stage 2 → comparison)
simrag experiment run              # ~6-8 hours

# Individual components
simrag experiment baseline         # Baseline RAG test (~2-3 minutes)
simrag experiment simrag           # SimRAG training only (~6-8 hours)
simrag experiment compare          # Compare results
simrag experiment test             # Interactive model testing

# With options
simrag experiment run --test-data         # Use test data instead of Alpaca
simrag experiment run --skip-baseline     # Skip baseline, run only SimRAG
simrag experiment baseline -d /path/to/docs  # Custom documents path
```

### Configuration
```bash
simrag config                      # Display current configuration
```

### Code Quality
```bash
black simrag_reproduction/         # Format code (line-length: 120)
ruff check simrag_reproduction/    # Lint
mypy simrag_reproduction/          # Type checking
```

## Architecture Overview

### Core Systems

**1. AI Providers Gateway** ([simrag_reproduction/ai_providers/](simrag_reproduction/ai_providers/))
- Multi-provider abstraction for LLM access
- `AIGateway` routes requests to available providers (Purdue API, HuggingFace)
- Purdue API is preferred for synthetic QA generation during training (faster)
- HuggingFace is used for baseline testing and local model testing

**2. RAG System** ([simrag_reproduction/rag/](simrag_reproduction/rag/))
- `BasicRAG`: Main orchestrator for document indexing and querying
- `DocumentRetriever`: Creates embeddings using `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- `VectorStore`: Wraps Qdrant (supports in-memory or persistent at `data/qdrant_db/`)
- `DocumentIngester`: File processing and chunking

**3. SimRAG Pipeline** ([simrag_reproduction/simrag/](simrag_reproduction/simrag/))
- **Stage 1** (`instruction_following.py`): Fine-tunes on Alpaca instruction-following dataset (52K examples)
- **Stage 2** (`domain_adaptation.py`): Fine-tunes on synthetic QA pairs from domain documents, with optional self-improvement loop
- `SyntheticQAGeneration`: Generates QA pairs from documents using LLM
- All stages inherit from `SimRAGBase` for common functionality

**4. Model Tuning** ([simrag_reproduction/tuning/](simrag_reproduction/tuning/))
- `BasicTuner`: QLoRA training orchestrator (4-bit quantization, LoRA adapters)
- `ModelRegistry`: Tracks all model versions with metadata linking stages via `experiment_run_id`
- Models can be tested locally using the `test` command or exported for Colab

**5. Experiments** ([simrag_reproduction/experiments/](simrag_reproduction/experiments/))
- `run_experiment.py`: Main orchestrator for complete pipeline
- `baseline/run_baseline.py`: Tests vanilla RAG performance
- `simrag/run_stage1.py`, `run_stage2.py`, `run_full_pipeline.py`: Stage orchestration
- `comparison/compare_results.py`: Performance analysis and visualization
- Results saved with timestamps to `experiments/{baseline,simrag}/results/`

### Configuration System

Configuration uses dataclasses with environment variable overrides ([simrag_reproduction/config.py](simrag_reproduction/config.py)):

- `RAGConfig`: Model selection, provider choice, vector store settings, generation parameters
- `TuningConfig`: Training hyperparameters, QLoRA settings, SimRAG-specific options

Key environment variables (all optional - project has sensible defaults):
```bash
MODEL_SIZE=small|medium           # small=Qwen2.5-1.5B, medium=Qwen2.5-7B
PURDUE_API_KEY=your-key           # Optional - for faster QA generation (fallback to HuggingFace)
TUNING_BATCH_SIZE=1-16            # Training batch size
TUNING_EPOCHS=1-10                # Training epochs
LORA_R=8-64                       # LoRA rank (default: 16)
LORA_ALPHA=1-128                  # LoRA scaling (default: 32)
SIMRAG_IMPROVEMENT_ROUNDS=1-10    # Stage 2 self-improvement rounds (default: 1)
RANDOM_SEED=42                    # Reproducibility seed
```

Copy `.env.example` to `.env` for customization.

### Data Flow

**Baseline Experiment:**
```
Documents → DocumentIngester → VectorStore (Qdrant)
→ Query → DocumentRetriever (semantic search)
→ AIGateway.chat() → Answer
```

**SimRAG Training:**
```
Stage 1: Alpaca dataset → BasicTuner (QLoRA) → Stage 1 model → ModelRegistry
         → ModelRegistry (version tracking)

Stage 2: Domain documents → SyntheticQAGeneration (via AIGateway)
         → QA pairs → BasicTuner (QLoRA) → Stage 2 model → ModelRegistry
         → ModelRegistry (version tracking)

Self-improvement: Repeat Stage 2 using previous round's model for QA generation
```

### Model Versioning & Storage

Models stored in `tuned_models/model_{1b,8b}/{stage_1,stage_2}/v{X.Y}/`:
- LoRA adapters (~100MB for 1.5B, ~400MB for 7B)
- Metadata tracked in `model_registry.json` with `experiment_run_id` linking stages
- Tested locally using `simrag experiment test` or exported for Colab

## Important Implementation Details

### Reproducibility
- **Random seeds** set to 42 (configurable via `RANDOM_SEED`)
- **cuDNN deterministic mode NOT used** - causes 10x slowdown with minimal reproducibility benefit
- All results include metadata for validation

### Performance Optimizations
- **Gradient checkpointing** enabled to reduce memory usage during training
- **HuggingFace** is used for all local testing and inference
- **QLoRA (4-bit quantization)** reduces VRAM to ~3-4GB for 1.5B model training

### Provider Selection
- `PURDUE_API_KEY`: Optional - for faster QA generation during training (fallback to HuggingFace)
- All testing uses HuggingFace directly (no external services required)

### Self-Improvement Loop
- Controlled by `SIMRAG_IMPROVEMENT_ROUNDS` (default: 1 = no self-improvement)
- Each round uses the improved model from previous round for better QA generation
- All rounds linked via `experiment_run_id` in model registry

### Hardware Requirements
Expected runtimes on RTX 3080 (10GB VRAM):
- Baseline: 2-3 minutes
- Stage 1: 3-4 hours (52K examples)
- Stage 2: 3-4 hours per round
- Full pipeline: 6-8 hours

Memory usage with QLoRA:
- Training: ~3-4GB VRAM (1.5B), ~6-7GB VRAM (7B)
- Inference: ~1.5GB VRAM (1.5B)

## Common Development Patterns

### Adding a New Test
Add to appropriate test directory (`tests/tests_{category}/`), following pytest conventions:
- Files: `test_*.py` or `*_test.py`
- Classes: `Test*`
- Functions: `test_*`

### Adding a New AI Provider
1. Extend `BaseLLMClient` in `simrag_reproduction/ai_providers/base_client.py`
2. Implement `chat()` and `generate()` methods
3. Register in `AIGateway._setup_providers()`
4. Add environment variable support in `config.py`

### Running Experiments Programmatically
```python
from simrag_reproduction.experiments.run_experiment import run_complete_experiment

results = run_complete_experiment(
    documents_folder="data/documents",
    use_real_datasets=True,  # True=Alpaca, False=test data
    skip_baseline=False,
    skip_simrag=False
)
```

### Model Testing
Test specific trained models interactively:
```bash
simrag experiment test  # Interactive prompts for stage/model selection
```

## Debugging Tips

- **Purdue API issues**: If `PURDUE_API_KEY` is not set, the system will use HuggingFace (slower but works offline)
- **CUDA OOM**: Reduce `TUNING_BATCH_SIZE`, switch to smaller model
- **Model not loading**: Check `HF_TOKEN` for gated models (Llama), verify internet connection
- **Vector store issues**: Delete `data/qdrant_db/` or set `USE_PERSISTENT=false` for in-memory mode
- **Poetry environment**: Always run `poetry shell` before commands, or use `poetry run simrag`
- **Model running on CPU instead of GPU**: The test command uses HuggingFace with 4-bit quantization. Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## File Locations for Common Tasks

| Task | File(s) |
|------|---------|
| Configure system | [simrag_reproduction/config.py](simrag_reproduction/config.py), `.env` |
| Run RAG | [simrag_reproduction/rag/rag_setup.py](simrag_reproduction/rag/rag_setup.py) |
| Fine-tune model | [simrag_reproduction/tuning/basic_tuning.py](simrag_reproduction/tuning/basic_tuning.py) |
| Stage 1 training | [simrag_reproduction/simrag/instruction_following.py](simrag_reproduction/simrag/instruction_following.py) |
| Stage 2 training | [simrag_reproduction/simrag/domain_adaptation.py](simrag_reproduction/simrag/domain_adaptation.py) |
| Pipeline orchestration | [simrag_reproduction/experiments/run_experiment.py](simrag_reproduction/experiments/run_experiment.py) |
| CLI interface | [simrag_reproduction/cli/main.py](simrag_reproduction/cli/main.py) |
| Logging setup | [simrag_reproduction/logging_config.py](simrag_reproduction/logging_config.py) |

## Dependencies

- **Python**: 3.12 (required for PyTorch CUDA 12.1 support)
- **PyTorch**: >=2.0.0 from CUDA 12.1 index
- **Key libraries**: transformers, datasets, sentence-transformers, qdrant-client, bitsandbytes, peft, accelerate
- **Optional external services**: Purdue GenAI API (for faster QA generation, fallback to HuggingFace)

See [pyproject.toml](pyproject.toml) for complete dependency list.
