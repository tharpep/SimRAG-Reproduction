# Model Usage Guide

## Critical: Fine-Tuned Models vs Base Models

**IMPORTANT**: Fine-tuned models are saved in HuggingFace format, NOT in Ollama format. Ollama only knows about models in its own library (base models like "llama3.2:1b").

## How Models Are Used

### 1. **Baseline Testing** → Ollama (Base Model)
- **Provider**: Ollama
- **Model**: Base model from Ollama library (e.g., "llama3.2:1b")
- **Why**: Testing baseline performance before fine-tuning
- **Location**: `experiments/baseline/run_baseline.py`

### 2. **Synthetic QA Generation** → Purdue API
- **Provider**: Purdue GenAI API
- **Model**: `llama4:latest` (default, independent of model_size config)
- **Why**: Generating training data for intermediate steps, not testing model performance
- **Location**: `simrag/synthetic_qa_generation.py`
- **Note**: Purdue API model selection is independent of `model_size` config. It uses its own default model names (e.g., "llama4:latest", "llama3.1:latest", "llama3.1:70b", "mistral:latest")

### 3. **SimRAG Testing** → HuggingFace (Fine-Tuned Model)
- **Provider**: HuggingFace Transformers (direct loading)
- **Model**: Fine-tuned model from `./tuned_models/llama_1b/vX.X/`
- **Why**: Testing the actual fine-tuned model performance
- **Location**: `experiments/simrag/run_full_pipeline.py`

## Model Loading Flow

### Training Flow:
1. **Stage 1 Training**: Loads base model from HuggingFace → Fine-tunes → Saves to `./tuned_models/llama_1b/stage_1/v1.0/`
2. **Stage 2 Training**: Loads base model from HuggingFace → Fine-tunes → Saves to `./tuned_models/llama_1b/stage_2/v1.0/`

### Testing Flow:
1. **Baseline**: Uses Ollama with base model "llama3.2:1b"
2. **SimRAG**: Loads fine-tuned model from `./tuned_models/llama_1b/stage_2/v1.0/` using HuggingFace Transformers

## Technical Details

### HuggingFace Client
- **File**: `ai_providers/huggingface_client.py`
- **Purpose**: Load fine-tuned models directly from disk
- **Usage**: Automatically used when `model_path` is provided to `BasicRAG`

### Model Registry
- **File**: `tuning/model_registry.py`
- **Purpose**: Tracks fine-tuned model versions
- **Location**: `./tuned_models/llama_1b/model_registry.json`

### How to Verify Correct Model is Used

1. **Check logs**: Look for "Using fine-tuned model: /path/to/model"
2. **Check model path**: Fine-tuned models are in `./tuned_models/llama_1b/vX.X/`
3. **Baseline vs SimRAG**: Baseline uses Ollama (base model), SimRAG uses HuggingFace (fine-tuned)

## Common Issues

### ❌ Wrong: Using Ollama for Fine-Tuned Models
- Ollama doesn't know about fine-tuned models
- It will always use the base model from its library

### ✅ Correct: Using HuggingFace for Fine-Tuned Models
- Loads the actual fine-tuned model from disk
- Uses the exact model that was trained

## Code Examples

### Baseline (Base Model via Ollama)
```python
rag = BasicRAG(force_provider="ollama")  # Uses base model
```

### SimRAG (Fine-Tuned Model via HuggingFace)
```python
model_path = "./tuned_models/llama_1b/v1.0"
rag = BasicRAG(model_path=model_path)  # Uses fine-tuned model
```

### Synthetic QA (Purdue API)
```python
rag = BasicRAG(force_provider="purdue")  # Uses Purdue's default model (llama4:latest)
# Purdue model selection is independent of model_size config
# Available Purdue models: llama4:latest, llama3.1:latest, llama3.1:70b, mistral:latest, mixtral:latest
```

