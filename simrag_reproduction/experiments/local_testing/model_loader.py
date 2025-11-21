"""
Model Loader
Handles loading baseline and fine-tuned models with 4-bit quantization
"""

import gc
from typing import Tuple, Any

# Initialize IMPORT_ERROR at module level to avoid undefined variable errors
IMPORT_ERROR = ""

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    IMPORT_ERROR = str(e)

from ...logging_config import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Model loader for baseline and fine-tuned models
    Uses 4-bit quantization for efficient memory usage
    """
    
    @staticmethod
    def load_baseline_model(base_model_name: str) -> Tuple[Any, Any]:
        """
        Load baseline model with 4-bit quantization
        
        Args:
            base_model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if not DEPS_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {IMPORT_ERROR}")
        
        logger.info("Loading baseline model (this will take 1-2 minutes)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"✓ Baseline model loaded (VRAM: {vram_used:.2f} GB)")
        else:
            logger.info(f"✓ Baseline model loaded (CPU mode)")
        
        return model, tokenizer
    
    @staticmethod
    def load_finetuned_model(adapter_path: str, base_model_name: str) -> Tuple[Any, Any]:
        """
        Load fine-tuned model (base + PEFT adapters) with 4-bit quantization
        
        Args:
            adapter_path: Path to LoRA adapter directory
            base_model_name: HuggingFace model name
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if not DEPS_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {IMPORT_ERROR}")
        
        logger.info("Loading fine-tuned model (this will take 1-2 minutes)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"✓ Fine-tuned model loaded (VRAM: {vram_used:.2f} GB)")
        else:
            logger.info(f"✓ Fine-tuned model loaded (CPU mode)")
        
        return model, tokenizer
    
    @staticmethod
    def cleanup_model(model: Any, tokenizer: Any) -> None:
        """
        Clean up model and free VRAM
        
        Args:
            model: Model to cleanup
            tokenizer: Tokenizer to cleanup
        """
        logger.info("Cleaning up model...")
        
        del model
        del tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        logger.info("✓ Model unloaded, VRAM freed")

