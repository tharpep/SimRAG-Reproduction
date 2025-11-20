# Tuned Models

This directory contains fine-tuned model versions organized by model size and training stage.

**For detailed documentation, see the main [README.md](../README.md#model-management).**

## Quick Reference

- `model_1b/` - Small model (1.5B parameters)
- `model_8b/` - Medium model (7B parameters)
- Each contains `stage_1/` and `stage_2/` subdirectories with versioned models
- Models are saved as LoRA adapters (~100MB for 1.5B, ~400MB for 7B)
- `model_registry.json` tracks all versions and metadata

Models are automatically registered with Ollama after training for fast inference.
