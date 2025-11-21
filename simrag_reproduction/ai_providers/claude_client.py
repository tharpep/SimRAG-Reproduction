"""
Anthropic Claude API Client
Basic chat functionality for synthetic QA generation
"""

import os
from typing import Optional, List, Any
from .base_client import BaseLLMClient

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except (OSError, IOError) as e:
            # Silently fail - environment variables may be set elsewhere
            pass

load_env_file()


class ClaudeClient(BaseLLMClient):
    """Client for Anthropic Claude API"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Claude client
        
        Args:
            api_key: API key for Anthropic Claude. If None, will try to load from CLAUDE_API_KEY environment variable
            model: Model to use (default: claude-haiku-4-5-20251001)
            
        Raises:
            ValueError: If API key is not provided or found
            ImportError: If anthropic package is not installed
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Claude support. "
                "Install it with: pip install anthropic"
            )
        
        self.api_key = api_key or os.getenv('CLAUDE_API_KEY')
        if not self.api_key or not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("API key is required. Provide it directly or set CLAUDE_API_KEY environment variable.")
        
        self.model = model or "claude-haiku-4-5-20251001"
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def chat(self, messages: Any, model: Optional[str] = None, **kwargs) -> str:
        """
        Send a message and get a response
        
        Args:
            messages: Your message (str) or messages list
            model: Model to use (uses default if not specified)
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            str: AI response
            
        Raises:
            ValueError: If messages is invalid or response parsing fails
            Exception: If API call fails
        """
        if not messages:
            raise ValueError("messages cannot be empty")
        
        # Use specified model or default
        model_name = model or self.model
        
        # Handle both string and message list formats
        if isinstance(messages, str):
            messages_list = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            # Convert to Claude format if needed
            messages_list = []
            for msg in messages:
                if isinstance(msg, dict):
                    # Ensure role is 'user' or 'assistant'
                    role = msg.get("role", "user")
                    if role not in ["user", "assistant"]:
                        role = "user"
                    messages_list.append({
                        "role": role,
                        "content": msg.get("content", str(msg))
                    })
                else:
                    messages_list.append({"role": "user", "content": str(msg)})
        else:
            messages_list = [{"role": "user", "content": str(messages)}]
        
        try:
            # Extract generation parameters
            max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 1024))
            temperature = kwargs.get("temperature", 0.7)
            
            # Claude API call
            response = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages_list
            )
            
            # Extract text from response
            if response.content and len(response.content) > 0:
                # Claude returns a list of content blocks
                text_parts = []
                for block in response.content:
                    if hasattr(block, 'text'):
                        text_parts.append(block.text)
                    elif isinstance(block, str):
                        text_parts.append(block)
                
                if text_parts:
                    return "\n".join(text_parts)
                else:
                    raise ValueError("Empty response from Claude API")
            else:
                raise ValueError("No content in Claude API response")
                
        except Exception as e:
            # Re-raise with more context
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise ValueError(f"Claude API authentication failed: {error_msg}")
            elif "rate_limit" in error_msg.lower():
                raise Exception(f"Claude API rate limit exceeded: {error_msg}")
            else:
                raise Exception(f"Error calling Claude API: {error_msg}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Claude models"""
        return [
            "claude-haiku-4-5-20251001"
        ]

