"""
SimRAG Implementation
Self-improving Retrieval-Augmented Generation
"""

__version__ = "1.0.0"
__author__ = "ECE 570 Student"

# Import main SimRAG components
from .stage1.instruction_tuner import SimRAGStageI
from .stage2.synthetic_qa_generator import SyntheticQAGenerator
from .stage2.domain_tuner import SimRAGStageII
from .evaluation.performance_comparison import PerformanceComparator

__all__ = [
    "SimRAGStageI",
    "SyntheticQAGenerator", 
    "SimRAGStageII",
    "PerformanceComparator"
]
