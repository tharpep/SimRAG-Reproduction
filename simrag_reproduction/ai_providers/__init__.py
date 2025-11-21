from .base_client import BaseLLMClient
from .gateway import AIGateway
from .purdue_api import PurdueGenAI
from .huggingface_client import HuggingFaceClient

# Try to import ClaudeClient (optional dependency)
try:
    from .claude_client import ClaudeClient
    __all__ = ['BaseLLMClient', 'AIGateway', 'PurdueGenAI', 'HuggingFaceClient', 'ClaudeClient']
except ImportError:
    __all__ = ['BaseLLMClient', 'AIGateway', 'PurdueGenAI', 'HuggingFaceClient']

