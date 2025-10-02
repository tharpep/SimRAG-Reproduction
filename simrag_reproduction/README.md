# SimRAG Reproduction

A simplified implementation of the SimRAG paper for learning RAG and fine-tuning concepts.

## Overview

This codebase provides a clean, educational implementation focusing on:

- **AI Providers**: Unified interface for Ollama (local) and Purdue GenAI
- **RAG System**: Basic retrieval-augmented generation with Qdrant
- **Testing**: Comprehensive test suite for all components

## Quick Start

```bash
# Setup environment
python run setup

# Run tests
python run test

# Try the RAG demo
python -m rag.basic_rag
```

## Architecture

```
simrag_reproduction/
├── ai_providers/          # LLM client abstractions
│   ├── gateway.py         # Main AI gateway
│   ├── local.py          # Ollama client
│   ├── purdue_api.py     # Purdue GenAI client
│   └── base_client.py    # Abstract base class
├── rag/                   # RAG implementations
│   └── basic_rag.py      # Basic RAG with Qdrant
├── tests/                 # Test suite
├── run                    # CLI tool
└── requirements.txt       # Dependencies
```

## Configuration

Create a `.env` file:

```bash
# For Ollama (local)
USE_OLLAMA=true
MODEL_NAME=qwen3:1.7b

# OR for Purdue GenAI
PURDUE_API_KEY=your-key-here
```

## Components

### AI Gateway
Unified interface for different LLM providers:
- Auto-selects available provider
- Handles async/sync calls
- Easy to extend with new providers

### RAG System
Basic retrieval-augmented generation:
- Sentence transformers for embeddings
- Qdrant for vector storage
- Context-aware question answering

### Testing
Comprehensive test coverage:
- Unit tests for all providers
- Mocked external dependencies
- Async test support

## Development

This is a learning-focused codebase. The implementation is intentionally simple to understand core concepts before building more complex features.

## Document Formatting Tips

For best RAG performance, structure your `.txt` and `.md` files:

### Content Structure (Critical)
- **Clear section headers** (`#`, `##`, `###`) - helps chunking at logical boundaries
- **Complete thoughts** (200-500 words per concept) - optimal chunk size
- **Consistent terminology** - improves semantic similarity matching

### Chunk Size Optimization
- **Current setting**: 1000 characters per chunk
- **Sweet spot**: 200-500 words per "concept" or "section"
- **Why it matters**: Too small = fragmented context, too large = diluted relevance

### Semantic Clarity (Key for Retrieval)
- **Write complete sentences** that stand alone
- **Use descriptive language** - "machine learning algorithms" vs "ML stuff"
- **Include context** - "The recommendation system achieved 95% accuracy" vs "95% accuracy"

### Example Format
```markdown
# Project Alpha
**Goal**: Build recommendation system
**Status**: In progress

## Key Findings
- Collaborative filtering works best for our use case
- Need at least 1000 user interactions for good results
- Matrix factorization outperforms neural networks
```

### What Doesn't Matter
- Markdown formatting (headers, bold, etc.) - gets stripped
- Line breaks - normalized to spaces
- File organization - each file becomes separate chunks

## Hardware Notes

- RTX 3080 10GB VRAM
- Supports both local (Ollama) and cloud (Purdue) LLMs
- Qdrant can run in-memory or persistent mode