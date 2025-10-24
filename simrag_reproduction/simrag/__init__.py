"""
SimRAG Implementation
Self-improving Retrieval-Augmented Generation
"""

__version__ = "1.0.0"
__author__ = "ECE 570 Student"

# Import main SimRAG components
from .base import SimRAGBase
from .instruction_following import InstructionFollowing
from .synthetic_qa_generation import SyntheticQAGeneration
from .domain_adaptation import DomainAdaptation

__all__ = [
    "SimRAGBase",
    "InstructionFollowing", 
    "SyntheticQAGeneration",
    "DomainAdaptation"
]
