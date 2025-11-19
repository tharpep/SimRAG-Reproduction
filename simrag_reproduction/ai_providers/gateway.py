"""
Simple AI Gateway
Routes requests to AI providers (Purdue GenAI Studio, Local Ollama)
Designed to be easily extended for additional providers
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from .purdue_api import PurdueGenAI
from .local import OllamaClient, OllamaConfig
from .huggingface_client import HuggingFaceClient
from ..config import get_rag_config

# Load environment variables from .env file
load_dotenv()


class AIGateway:
    """Simple gateway for AI requests"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize gateway with configuration
        
        Args:
            config: Dictionary with provider configurations
                   If None, will try to load from environment variables and config.py
        """
        self.providers = {}
        self.rag_config = get_rag_config()
        self._setup_providers(config or {})
    
    def _setup_providers(self, config: Dict[str, Any]):
        """Setup available AI providers"""
        # Setup Purdue provider
        if "purdue" in config:
            api_key = config["purdue"].get("api_key")
            self.providers["purdue"] = PurdueGenAI(api_key)
        elif os.getenv('PURDUE_API_KEY'):
            self.providers["purdue"] = PurdueGenAI()
        
        # Setup HuggingFace provider if model path is provided
        if "huggingface" in config:
            model_path = config["huggingface"].get("model_path")
            if model_path:
                self.providers["huggingface"] = HuggingFaceClient(model_path)
        
        # Setup Local Ollama provider
        if "ollama" in config:
            ollama_config = OllamaConfig(
                base_url=config["ollama"].get("base_url", "http://localhost:11434"),
                default_model=config["ollama"].get("default_model", self.rag_config.model_name)
            )
            self.providers["ollama"] = OllamaClient(ollama_config)
        elif self.rag_config.use_ollama or os.getenv('USE_OLLAMA', 'false').lower() == 'true':
            ollama_config = OllamaConfig(
                default_model=self.rag_config.model_name
            )
            self.providers["ollama"] = OllamaClient(ollama_config)
    
    def chat(self, message: str, provider: Optional[str] = None, model: Optional[str] = None, force_provider: bool = False) -> str:
        """
        Send a chat message to specified AI provider
        
        Args:
            message: Your message to the AI
            provider: AI provider to use ("purdue" or "ollama")
                     If None, auto-selects based on availability
            model: Model to use (uses provider default if not specified)
            force_provider: If True, raises error if provider not available (default: False)
            
        Returns:
            str: AI response
        """
        # Auto-select provider based on config if not specified
        if provider is None:
            if self.rag_config.use_ollama and "ollama" in self.providers:
                provider = "ollama"
            elif not self.rag_config.use_ollama and "purdue" in self.providers:
                provider = "purdue"
            elif "ollama" in self.providers:
                provider = "ollama"
            elif "purdue" in self.providers:
                provider = "purdue"
            else:
                raise Exception("No providers available. Set PURDUE_API_KEY or USE_OLLAMA=true")
        
        # Check if provider is available
        if provider not in self.providers:
            available = ", ".join(self.providers.keys())
            if force_provider:
                raise Exception(f"Provider '{provider}' not available. Available: {available}")
            else:
                # Fallback to available provider
                if "purdue" in self.providers:
                    provider = "purdue"
                elif "ollama" in self.providers:
                    provider = "ollama"
                else:
                    raise Exception(f"Provider '{provider}' not available. Available: {available}")
        
        provider_client = self.providers[provider]
        
        # Handle different provider types
        if provider == "ollama":
            return self._chat_ollama(provider_client, message, model)
        elif provider == "huggingface":
            # HuggingFace client ignores model parameter (uses loaded model)
            return provider_client.chat(message)
        else:
            # Use config model for Purdue API if no model specified
            model = model or self.rag_config.model_name
            return provider_client.chat(message, model)
    
    def _chat_ollama(self, client: OllamaClient, message: str, model: Optional[str] = None) -> str:
        """Helper to handle Ollama calls"""
        # Now that OllamaClient.chat() is synchronous, we can call it directly
        return client.chat(message, model=model)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
