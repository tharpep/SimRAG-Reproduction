"""
HuggingFace Transformers Client
For loading and using fine-tuned models directly (bypassing Ollama)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import logging
from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseLLMClient):
    """Client for loading and using fine-tuned HuggingFace models directly"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize HuggingFace client with a fine-tuned model
        
        Args:
            model_path: Path to fine-tuned model directory (HuggingFace format)
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
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info(f"Loading fine-tuned model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" and self.device != "mps":
                self.model = self.model.to(self.device)
            
            logger.info(f"Fine-tuned model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            raise
    
    def chat(self, messages: Any, model: Optional[str] = None, **kwargs) -> str:
        """
        Generate response using the fine-tuned model
        
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
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generation parameters
        max_new_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 200))
        temperature = kwargs.get("temperature", 0.7)
        do_sample = temperature > 0
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def get_available_models(self) -> list:
        """Return list with just the loaded model path"""
        return [self.model_path]

