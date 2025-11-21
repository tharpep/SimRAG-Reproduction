"""
HuggingFace Transformers Client
For loading and using HuggingFace models (from Hub or local paths)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional, Dict, Any
import logging
import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from .base_client import BaseLLMClient

# Load environment variables (for HF_TOKEN)
load_dotenv()

# Constants for magic numbers
DEFAULT_MAX_TOKENS = 200
TIMEOUT_SECONDS_PER_100_TOKENS = 30
MIN_TIMEOUT_SECONDS = 60

# Set HuggingFace token from environment if available
# Transformers library automatically uses HF_TOKEN or HUGGINGFACE_HUB_TOKEN
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = hf_token

logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseLLMClient):
    """
    Client for loading and using HuggingFace models
    
    Supports both:
    - HuggingFace Hub model IDs (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
    - Local model paths (e.g., "./tuned_models/model_1b/stage_1/v1.0/")
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize HuggingFace client
        
        Args:
            model_path: HuggingFace Hub model ID or local path to model directory
            device: Device to use ("auto", "cuda", "cpu")
            
        Raises:
            ValueError: If model_path is invalid
        """
        if not model_path or not isinstance(model_path, str) or not model_path.strip():
            raise ValueError(f"model_path must be a non-empty string, got: {model_path}")
        
        self.model_path = model_path
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return device
    
    def _is_local_path(self, path: str) -> bool:
        """Check if path is a local directory (vs HuggingFace Hub ID)"""
        return os.path.exists(path) and os.path.isdir(path)
    
    def _load_model(self):
        """Load the model and tokenizer (from Hub or local path)"""
        try:
            # Determine if this is a local path or Hub ID
            is_local = self._is_local_path(self.model_path)
            
            # Check if this is a LoRA adapter (has adapter_config.json)
            adapter_config_path = os.path.join(self.model_path, "adapter_config.json") if is_local else None
            is_lora_adapter = adapter_config_path and os.path.exists(adapter_config_path)
            
            if is_local:
                logger.debug(f"Loading model from local path: {self.model_path}")
                if is_lora_adapter:
                    logger.debug("Detected LoRA adapter model")
            else:
                logger.debug(f"Loading model from HuggingFace Hub: {self.model_path}")
            
            # Load tokenizer (always from the adapter path or model path)
            logger.debug("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.debug(f"Loading model (device: {self.device})...")
            
            # Configure 4-bit quantization for CUDA
            quantization_config = None
            if self.device == "cuda":
                logger.debug("Configuring 4-bit quantization for efficient inference...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            if is_lora_adapter:
                # Load LoRA adapter model
                # 1. Load adapter config to get base model info
                try:
                    with open(adapter_config_path, 'r', encoding='utf-8') as f:
                        adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path")
                    if not base_model_name:
                        raise ValueError(f"Missing 'base_model_name_or_path' in adapter config: {adapter_config_path}")
                except (OSError, IOError) as e:
                    raise IOError(f"Failed to read adapter config file {adapter_config_path}: {e}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in adapter config file {adapter_config_path}: {e}")
                
                logger.debug(f"Loading base model: {base_model_name}")
                
                # 2. Load base model (with quantization if CUDA)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                
                # 3. Load LoRA adapters
                logger.debug(f"Loading LoRA adapters from {self.model_path}...")
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                
                # Move to device if not CUDA (CUDA uses device_map="auto")
                if self.device == "mps":
                    logger.debug(f"Moving model to {self.device}...")
                    self.model = self.model.to(self.device)
                elif self.device == "cpu" and quantization_config is None:
                    self.model = self.model.to(self.device)
                
                logger.debug(f"LoRA adapter model loaded on {self.device}")
                
            else:
                # Load full model (base or fully fine-tuned)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                
                # Move to device explicitly ONLY if NOT using quantization (quantization handles this)
                if self.device == "cuda" and quantization_config is None:
                    logger.debug(f"Moving model to {self.device}...")
                    self.model = self.model.to(self.device)
                elif self.device == "mps":
                    logger.debug(f"Moving model to {self.device}...")
                    self.model = self.model.to(self.device)
            
            if quantization_config:
                 logger.debug("4-bit quantization enabled: Model loaded efficiently")
            
            source = "local LoRA adapter" if is_lora_adapter else ("local path" if is_local else "HuggingFace Hub")
            logger.debug(f"Model loaded from {source}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def chat(self, messages: Any, model: Optional[str] = None, **kwargs) -> str:
        """
        Generate response using the loaded model
        
        Args:
            messages: Chat message (string or list of dicts)
            model: Ignored (uses loaded model)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
        
        # Convert string message to prompt
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list) and len(messages) > 0:
            # Extract text from message format
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                prompt = last_msg.get("content", "")
            else:
                prompt = str(last_msg)
        else:
            prompt = str(messages)
        
        # Tokenize input - use proper tokenizer method for chat models
        # For Instruct models, we should use apply_chat_template, but for simplicity
        # we'll use the prompt directly
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation parameters
        max_new_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", DEFAULT_MAX_TOKENS))
        temperature = kwargs.get("temperature", 0.7)
        do_sample = temperature > 0
        
        logger.info(f"Generating response (max_tokens={max_new_tokens}, device={self.device})...")
        
        # Clear GPU cache BEFORE generation to free any lingering memory
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()  # Wait for all operations to complete
            torch.cuda.empty_cache()  # Clear unused cache
        
        # Calculate timeout: N seconds per 100 tokens (reasonable for GPU)
        timeout_seconds = max(MIN_TIMEOUT_SECONDS, (max_new_tokens / 100) * TIMEOUT_SECONDS_PER_100_TOKENS)
        start_time = time.time()
        
        # Generate
        try:
            # Use inference_mode for better performance and memory efficiency
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,  # Ensure at least 1 token is generated
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Re-enable KV cache for speed (we clear cache between calls)
                    num_beams=1,  # Greedy decoding (faster than beam search)
                    repetition_penalty=1.1,  # Prevent repetition loops
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                )
                
                # Check for timeout (safety check)
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    logger.warning(f"Generation took {elapsed:.1f}s (timeout: {timeout_seconds}s)")
            
            # Decode output (skip input tokens)
            if not outputs or len(outputs) == 0:
                raise ValueError("Model generation returned empty output")
            if "input_ids" not in inputs or inputs["input_ids"].shape[0] == 0:
                raise ValueError("Invalid input_ids shape")
            
            input_length = inputs["input_ids"].shape[1]
            if len(outputs[0]) <= input_length:
                raise ValueError(f"Generated output length ({len(outputs[0])}) is not greater than input length ({input_length})")
            
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clear GPU cache after generation to prevent memory buildup
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()  # Wait for all operations to complete
                torch.cuda.empty_cache()  # Clear unused cache
            
            logger.debug(f"Generated {len(generated_tokens)} tokens")
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            # Clear cache even on error
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def get_available_models(self) -> list:
        """Return list with just the loaded model path"""
        return [self.model_path]

