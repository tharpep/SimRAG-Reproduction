"""
HuggingFace Transformers Client
For loading and using HuggingFace models (from Hub or local paths)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from .base_client import BaseLLMClient

# Load environment variables (for HF_TOKEN)
load_dotenv()

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
        """
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
            
            if is_local:
                logger.info(f"Loading model from local path: {self.model_path}")
            else:
                logger.info(f"Loading model from HuggingFace Hub: {self.model_path} (this may take a moment on first download)")
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loading model (device: {self.device})...")
            # Load model - use float16 for inference (faster, less memory)
            # For training, model should be loaded in float32 (handled separately)
            # Note: This may take several minutes on first download
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None,
                low_cpu_mem_usage=True  # More efficient memory usage
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" and self.device != "mps":
                logger.info(f"Moving model to {self.device}...")
                self.model = self.model.to(self.device)
                logger.info("Model moved to device")
            
            source = "local path" if is_local else "HuggingFace Hub"
            logger.info(f"Model loaded from {source} on {self.device}")
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
        max_new_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 200))
        temperature = kwargs.get("temperature", 0.7)
        do_sample = temperature > 0
        
        logger.info(f"Generating response (max_tokens={max_new_tokens}, device={self.device})...")
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode output (skip input tokens)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            logger.info(f"Generated {len(generated_tokens)} tokens")
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def get_available_models(self) -> list:
        """Return list with just the loaded model path"""
        return [self.model_path]

