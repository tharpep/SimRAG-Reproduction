"""
SimRAG Reproduction Package
A simplified implementation of the SimRAG paper for learning RAG and fine-tuning concepts.
"""

from .config import get_rag_config, get_tuning_config, RAGConfig, TuningConfig
from .ai_providers import AIGateway
from .logging_config import get_logger, setup_logging

__all__ = [
    "get_rag_config",
    "get_tuning_config",
    "RAGConfig",
    "TuningConfig",
    "AIGateway",
    "get_logger",
    "setup_logging",
]

__version__ = "0.1.0"

