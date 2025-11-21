"""
Simple AI Gateway
Routes requests to AI providers (Claude, Purdue GenAI Studio, HuggingFace)
Designed to be easily extended for additional providers
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from .purdue_api import PurdueGenAI
from .huggingface_client import HuggingFaceClient
from ..config import get_rag_config
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import Claude client (optional dependency)
try:
    from .claude_client import ClaudeClient
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


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
        # Setup Claude provider (for QA generation)
        if CLAUDE_AVAILABLE:
            if "claude" in config:
                api_key = config["claude"].get("api_key")
                model = config["claude"].get("model")
                try:
                    self.providers["claude"] = ClaudeClient(api_key, model)
                except Exception as e:
                    logger.warning(f"Failed to initialize Claude client: {e}")
            elif os.getenv('CLAUDE_API_KEY'):
                try:
                    self.providers["claude"] = ClaudeClient()
                except Exception as e:
                    logger.warning(f"Failed to initialize Claude client: {e}")
        
        # Setup Purdue provider (for QA generation)
        if "purdue" in config:
            api_key = config["purdue"].get("api_key")
            self.providers["purdue"] = PurdueGenAI(api_key)
        elif os.getenv('PURDUE_API_KEY'):
            self.providers["purdue"] = PurdueGenAI()
        
        # Setup HuggingFace provider (for inference when needed)
        if "huggingface" in config:
            model_path = config["huggingface"].get("model_path")
            if model_path:
                logger.debug(f"Initializing HuggingFace client with model: {model_path}")
                try:
                    self.providers["huggingface"] = HuggingFaceClient(model_path)
                except Exception as e:
                    logger.warning(f"Failed to load HuggingFace model {model_path}: {e}")
                    # Don't raise - allow other providers to be used
    
    def chat(self, message: str, provider: Optional[str] = None, model: Optional[str] = None, force_provider: bool = False, **kwargs) -> str:
        """
        Send a chat message to specified AI provider
        
        Args:
            message: Your message to the AI
            provider: AI provider to use ("purdue" or "huggingface")
                     If None, auto-selects based on availability
            model: Model to use (uses provider default if not specified)
            force_provider: If True, raises error if provider not available (default: False)
            **kwargs: Additional parameters to pass to provider (e.g., max_tokens, temperature)
            
        Returns:
            str: AI response
            
        Raises:
            ValueError: If message is empty or invalid
            RuntimeError: If no providers are available or provider call fails
        """
        if not message or not isinstance(message, str):
            raise ValueError("message must be a non-empty string")
        # Auto-select provider based on config if not specified
        if provider is None:
            # Priority: Use config preference, then claude > purdue > huggingface
            preferred_provider = self.rag_config.qa_provider if hasattr(self.rag_config, 'qa_provider') else None
            if preferred_provider and preferred_provider in self.providers:
                provider = preferred_provider
            elif "claude" in self.providers:
                provider = "claude"
            elif "purdue" in self.providers:
                provider = "purdue"
            elif "huggingface" in self.providers:
                provider = "huggingface"
            else:
                raise RuntimeError("No providers available. Please ensure at least one provider is configured.")
        
        # Check if provider is available
        if provider not in self.providers:
            available = ", ".join(self.providers.keys()) if self.providers else "none"
            if force_provider:
                raise RuntimeError(f"Provider '{provider}' not available. Available: {available}")
            else:
                # Fallback to available provider (priority: claude > purdue > huggingface)
                if "claude" in self.providers:
                    provider = "claude"
                elif "purdue" in self.providers:
                    provider = "purdue"
                elif "huggingface" in self.providers:
                    provider = "huggingface"
                else:
                    raise RuntimeError(f"Provider '{provider}' not available. Available: {available}")
        
        provider_client = self.providers[provider]
        
        # Handle different provider types
        # Don't wrap exceptions - let original exceptions propagate for better debugging
        if provider == "huggingface":
            # HuggingFace client ignores model parameter (uses loaded model)
            # Pass through kwargs (max_tokens, temperature, etc.)
            return provider_client.chat(message, **kwargs)
        elif provider == "claude":
            # Claude API supports model parameter and kwargs
            return provider_client.chat(message, model=model, **kwargs)
        else:
            # Purdue API uses its own default model ("llama4:latest") if not specified
            # Don't use config.model_name here since Purdue has different model names
            # Purdue is used for intermediate steps (QA generation), not model testing
            return provider_client.chat(message, model)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
