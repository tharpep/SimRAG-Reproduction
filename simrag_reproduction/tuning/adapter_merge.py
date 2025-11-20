"""
Merge LoRA Adapters into Base Model
Utility to merge trained LoRA adapters back into the base model for deployment
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def merge_lora_adapters(
    adapter_path: str,
    output_path: str,
    base_model_name: Optional[str] = None
) -> str:
    """
    Merge LoRA adapters into base model and save
    
    Args:
        adapter_path: Path to LoRA adapter directory
        output_path: Path to save merged model
        base_model_name: Base model name (if None, reads from adapter_config.json)
        
    Returns:
        Path to merged model
    """
    import json
    
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    
    logger.info(f"Merging LoRA adapters from {adapter_path}")
    
    # Read adapter config to get base model
    if base_model_name is None:
        adapter_config_path = adapter_path / "adapter_config.json"
        if not adapter_config_path.exists():
            raise ValueError(f"adapter_config.json not found in {adapter_path}")
        
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
        
        if not base_model_name:
            raise ValueError("Could not determine base model name from adapter config")
    
    logger.info(f"Loading base model: {base_model_name}")
    
    # Load base model (full precision for merging)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load on CPU to avoid VRAM issues
        low_cpu_mem_usage=True
    )
    
    logger.info(f"Loading LoRA adapters from {adapter_path}")
    
    # Load adapters
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    
    logger.info("Merging adapters into base model...")
    
    # Merge adapters into base weights
    merged_model = model.merge_and_unload()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving merged model to {output_path}")
    
    # Save merged model
    merged_model.save_pretrained(str(output_path))
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(output_path))
    
    logger.info(f"✓ Merged model saved to {output_path}")
    
    return str(output_path)


def main():
    """CLI entry point for merging adapters"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("adapter_path", help="Path to LoRA adapter directory")
    parser.add_argument("output_path", help="Path to save merged model")
    parser.add_argument("--base-model", help="Base model name (optional)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    merge_lora_adapters(
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        base_model_name=args.base_model
    )


if __name__ == "__main__":
    main()

