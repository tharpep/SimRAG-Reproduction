"""
Simple Purdue GenAI Studio API Client
Basic chat functionality for testing and verification
"""

import json
import os
import urllib.request
import urllib.error
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


class PurdueGenAI(BaseLLMClient):
    """Simple client for Purdue GenAI Studio"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Purdue GenAI client
        
        Args:
            api_key: API key for Purdue GenAI Studio. If None, will try to load from PURDUE_API_KEY environment variable
            
        Raises:
            ValueError: If API key is not provided or found
        """
        self.api_key = api_key or os.getenv('PURDUE_API_KEY')
        if not self.api_key or not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("API key is required. Provide it directly or set PURDUE_API_KEY environment variable.")
        self.base_url = "https://genai.rcac.purdue.edu/api/chat/completions"
    
    def chat(self, messages: Any, model: Optional[str] = None, **kwargs) -> str:
        """
        Send a message and get a response
        
        Args:
            messages: Your message (str) or messages list
            model: Model to use (default: llama4:latest)
            
        Returns:
            str: AI response
            
        Raises:
            ValueError: If messages is invalid or response parsing fails
            Exception: If API call fails
        """
        if not messages:
            raise ValueError("messages cannot be empty")
        # Use default model if none specified
        if model is None:
            model = "llama4:latest"
            
        # Handle both string and message list formats
        if isinstance(messages, list):
            messages = messages
        elif isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = [{"role": "user", "content": str(messages)}]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            body = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            
            # Make request
            data = json.dumps(body).encode('utf-8')
            req = urllib.request.Request(self.base_url, data=data, headers=headers, method='POST')
            
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    try:
                        response_data = json.loads(response.read().decode('utf-8'))
                        if "choices" not in response_data or not response_data["choices"]:
                            raise ValueError("Invalid response structure: missing 'choices'")
                        if "message" not in response_data["choices"][0] or "content" not in response_data["choices"][0]["message"]:
                            raise ValueError("Invalid response structure: missing 'message.content'")
                        return response_data["choices"][0]["message"]["content"]
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        raise ValueError(f"Failed to parse API response: {e}")
                else:
                    error_text = response.read().decode('utf-8')
                    raise Exception(f"API Error {response.status}: {error_text}")
                    
        except urllib.error.HTTPError as e:
            error_text = e.read().decode('utf-8') if hasattr(e, 'read') else str(e)
            raise Exception(f"HTTP Error {e.code}: {error_text}")
        except Exception as e:
            raise Exception(f"Error calling Purdue GenAI: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models (hardcoded for Purdue)"""
        return [
            "llama4:latest",
            "llama3.1:latest",
            "llama3.1:70b",
            "mistral:latest",
            "mixtral:latest"
        ]