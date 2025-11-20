"""
SimRAG Instruction Following
Stage 1: Fine-tune on instruction-following datasets
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..logging_config import get_logger
from .base import SimRAGBase

logger = get_logger(__name__)


class InstructionFollowing(SimRAGBase):
    """SimRAG Stage 1: Instruction-following fine-tuning"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", config: Optional[Any] = None):
        """
        Initialize instruction following trainer
        
        Args:
            model_name: Model to fine-tune
            config: TuningConfig instance (optional)
        """
        super().__init__(model_name, config)
        logger.info("Instruction Following trainer initialized")
    
    def load_instruction_datasets(self, dataset_names: Optional[List[str]] = None) -> List[str]:
        """
        Load instruction-following dataset from Hugging Face
        
        Uses Stanford Alpaca dataset (tatsu-lab/alpaca) - 52K instruction examples.
        Downloads and caches locally on first use.
        
        Args:
            dataset_names: Ignored - always uses Alpaca dataset
            
        Returns:
            List of instruction-following examples in "Question: ...\nAnswer: ..." format
            
        Raises:
            ImportError: If datasets library not installed
            Exception: If dataset loading fails
        """
        logger.info("Loading Alpaca instruction dataset from Hugging Face...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not found. Install with: pip install datasets")
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        try:
            # Load Alpaca dataset (downloads on first use, then uses cache)
            logger.info("Downloading/caching Alpaca dataset (this may take a moment on first run)...")
            dataset = load_dataset("tatsu-lab/alpaca", split="train")
            
            # Convert to instruction format
            examples = []
            for item in dataset:
                instruction = item.get("instruction", "").strip()
                input_text = item.get("input", "").strip()
                output = item.get("output", "").strip()
                
                # Format as Question/Answer (Alpaca format)
                if input_text:
                    # If there's input context, include it
                    formatted = f"Question: {instruction}\nInput: {input_text}\nAnswer: {output}"
                else:
                    # Simple instruction -> answer format
                    formatted = f"Question: {instruction}\nAnswer: {output}"
                
                examples.append(formatted)
            
            logger.info(f"Loaded {len(examples)} instruction examples from Alpaca dataset")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load Alpaca dataset: {e}")
            logger.warning("Falling back to test data")
            return self._generate_test_instruction_data()
    
    def _generate_test_instruction_data(self) -> List[str]:
        """Generate test instruction data for testing purposes"""
        return [
            "Question: What is Docker?\nAnswer: Docker is a platform for developing, shipping, and running applications in containers.",
            "Question: How does binary search work?\nAnswer: Binary search works by repeatedly dividing the search interval in half.",
            "Question: What is DevOps?\nAnswer: DevOps is a set of practices that combines software development and IT operations.",
            "Question: Explain Python programming.\nAnswer: Python is a high-level programming language known for its simple syntax and readability.",
            "Question: What is machine learning?\nAnswer: Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        ]
    
    def prepare_instruction_data(self, use_real_datasets: bool = True, 
                               dataset_names: Optional[List[str]] = None) -> List[str]:
        """
        Prepare instruction-following data for Stage 1
        
        Args:
            use_real_datasets: Whether to use Alpaca dataset (True) or test data (False)
            dataset_names: Ignored - always uses Alpaca dataset if use_real_datasets=True
            
        Returns:
            List of instruction-following examples
        """
        if use_real_datasets:
            return self.load_instruction_datasets()
        else:
            return self._generate_test_instruction_data()
    
    def train_stage_1(self, use_real_datasets: bool = True, 
                     dataset_names: Optional[List[str]] = None,
                     notes: str = "SimRAG Stage 1 - Instruction Following") -> Optional[Any]:
        """
        Train Stage 1: Instruction-following fine-tuning
        
        Args:
            use_real_datasets: Whether to use real datasets or test data
            dataset_names: List of dataset names to load
            notes: Training notes for versioning
            
        Returns:
            Model version object if successful
        """
        logger.info("=== SimRAG Stage 1: Instruction Following Training ===")
        
        # Prepare instruction data
        logger.info("Preparing instruction-following data...")
        instruction_data = self.prepare_instruction_data(use_real_datasets)
        
        # Load model and prepare training data
        self.load_model()
        train_dataset = self.prepare_training_data(instruction_data)
        
        # Setup trainer for Stage 1
        logger.info("Setting up trainer for Stage 1...")
        output_dir = self.config.get_stage_output_dir("stage_1")
        try:
            self.setup_trainer(
                train_dataset=train_dataset,
                output_dir=output_dir,
                num_epochs=self.config.simrag_stage_1_epochs,
                notes=notes
            )
        except Exception as e:
            logger.error(f"Failed to setup trainer for Stage 1: {e}")
            raise
        
        # Train Stage 1
        logger.info("Starting Stage 1 training...")
        try:
            version = self.train_model(notes)
            
            if version:
                logger.info(f"Stage 1 training completed! Version: {version.version}")
                logger.info(f"Training time: {version.training_time_seconds:.1f}s")
                if version.final_loss:
                    logger.info(f"Final loss: {version.final_loss:.4f}")
            else:
                logger.warning("Stage 1 training completed but no version returned")
            
            return version
        except Exception as e:
            logger.error(f"Stage 1 training failed: {e}")
            raise
    
    def test_stage_1_performance(self, rag_system: Any, test_questions: List[str]) -> Dict[str, Any]:
        """
        Test Stage 1 model performance on RAG tasks
        
        Args:
            rag_system: RAG system to test
            test_questions: List of test questions
            
        Returns:
            Performance metrics dictionary
        """
        logger.info("=== Testing Stage 1 Performance ===")
        return self.test_performance(rag_system, test_questions)


