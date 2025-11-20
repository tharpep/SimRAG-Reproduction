"""
Simple AI Gateway
Routes requests to AI providers (Purdue GenAI Studio, Local Ollama)
Designed to be easily extended for additional providers
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from .purdue_api import PurdueGenAI
from .local import OllamaClient, OllamaConfig
from .huggingface_client import HuggingFaceClient
from ..config import get_rag_config
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


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
        # Setup Purdue provider (for QA generation)
        if "purdue" in config:
            api_key = config["purdue"].get("api_key")
            self.providers["purdue"] = PurdueGenAI(api_key)
        elif os.getenv('PURDUE_API_KEY'):
            self.providers["purdue"] = PurdueGenAI()
        
        # Setup HuggingFace provider (only when explicitly requested)
        # Note: We no longer auto-initialize HuggingFace to avoid unnecessary model loading
        # HuggingFace is primarily used for training, not inference (Ollama is used for testing)
        if "huggingface" in config:
            model_path = config["huggingface"].get("model_path")
            if model_path:
                logger.debug(f"Initializing HuggingFace client with model: {model_path}")
                try:
                    self.providers["huggingface"] = HuggingFaceClient(model_path)
                except Exception as e:
                    logger.warning(f"Failed to load HuggingFace model {model_path}: {e}")
                    # Don't raise - allow other providers to be used
        
        # Setup Local Ollama provider (optional, only if explicitly requested)
        # Note: Ollama uses its own model names (e.g., "qwen2.5:1.5b"), not HuggingFace Hub IDs
        # If using Ollama, explicitly set the model name in config or use OLLAMA_MODEL env var
        if "ollama" in config:
            ollama_config = OllamaConfig(
                base_url=config["ollama"].get("base_url", "http://localhost:11434"),
                default_model=config["ollama"].get("default_model", os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"))
            )
            self.providers["ollama"] = OllamaClient(ollama_config)
        elif self.rag_config.use_ollama or os.getenv('USE_OLLAMA', 'false').lower() == 'true':
            # Use OLLAMA_MODEL env var or fallback to default Ollama model name
            ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
            ollama_config = OllamaConfig(
                default_model=ollama_model
            )
            try:
                self.providers["ollama"] = OllamaClient(ollama_config)
            except Exception as e:
                logger.warning(f"Failed to setup Ollama: {e}. Continuing without Ollama.")
    
    def chat(self, message: str, provider: Optional[str] = None, model: Optional[str] = None, force_provider: bool = False, **kwargs) -> str:
        """
        Send a chat message to specified AI provider
        
        Args:
            message: Your message to the AI
            provider: AI provider to use ("purdue" or "ollama")
                     If None, auto-selects based on availability
            model: Model to use (uses provider default if not specified)
            force_provider: If True, raises error if provider not available (default: False)
            **kwargs: Additional parameters to pass to provider (e.g., max_tokens, temperature)
            
        Returns:
            str: AI response
        """
        # Auto-select provider based on config if not specified
        if provider is None:
            # Priority: ollama (default for testing) > purdue > huggingface
            if "ollama" in self.providers:
                provider = "ollama"
            elif "purdue" in self.providers:
                provider = "purdue"
            elif "huggingface" in self.providers:
                provider = "huggingface"
            else:
                raise Exception("No providers available. Please ensure at least one provider is configured.")
        
        # Check if provider is available
        if provider not in self.providers:
            available = ", ".join(self.providers.keys())
            if force_provider:
                raise Exception(f"Provider '{provider}' not available. Available: {available}")
            else:
                # Fallback to available provider (priority: ollama > purdue > huggingface)
                if "ollama" in self.providers:
                    provider = "ollama"
                elif "purdue" in self.providers:
                    provider = "purdue"
                elif "huggingface" in self.providers:
                    provider = "huggingface"
                else:
                    raise Exception(f"Provider '{provider}' not available. Available: {available}")
        
        provider_client = self.providers[provider]
        
        # Handle different provider types
        if provider == "ollama":
            return self._chat_ollama(provider_client, message, model)
        elif provider == "huggingface":
            # HuggingFace client ignores model parameter (uses loaded model)
            # Pass through kwargs (max_tokens, temperature, etc.)
            return provider_client.chat(message, **kwargs)
        else:
            # Purdue API uses its own default model ("llama3.1:latest") if not specified
            # Don't use config.model_name here since Purdue has different model names
            # Purdue is used for intermediate steps (QA generation), not model testing
            return provider_client.chat(message, model)
    
    def _chat_ollama(self, client: OllamaClient, message: str, model: Optional[str] = None) -> str:
        """Helper to handle Ollama calls"""
        # Now that OllamaClient.chat() is synchronous, we can call it directly
        return client.chat(message, model=model)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
