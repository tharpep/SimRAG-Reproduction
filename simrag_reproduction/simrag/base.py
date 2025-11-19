"""
SimRAG Base Class
Common functionality for all SimRAG stages
"""

import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from logging_config import get_logger
from tuning.basic_tuning import BasicTuner
from tuning.model_registry import get_model_registry
from config import get_tuning_config
from rag.rag_setup import BasicRAG

logger = get_logger(__name__)


class SimRAGBase:
    """Base class for all SimRAG stages with common functionality"""
    
    def __init__(self, model_name: str = "llama3.2:1b", config: Optional[Any] = None):
        """
        Initialize SimRAG base functionality
        
        Args:
            model_name: Model to fine-tune
            config: TuningConfig instance (optional)
        """
        self.model_name = model_name
        try:
            self.config = config or get_tuning_config()
            self.tuner = BasicTuner(model_name, config=self.config)
            self.registry = get_model_registry(self.config)
            logger.info(f"SimRAG base initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SimRAG base: {e}")
            raise
    
    def load_model(self) -> None:
        """Load the model for training
        
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info("Loading model...")
            self.tuner.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_training_data(self, data: List[str], max_length: Optional[int] = None) -> Any:
        """
        Prepare training data
        
        Args:
            data: List of training examples
            max_length: Maximum sequence length
            
        Returns:
            Prepared training dataset
            
        Raises:
            ValueError: If data is empty or invalid
            Exception: If data preparation fails
        """
        if not data:
            raise ValueError("Training data cannot be empty")
        
        try:
            max_length = max_length or self.config.max_length
            logger.info(f"Preparing {len(data)} training examples with max_length={max_length}")
            dataset = self.tuner.prepare_data(data, max_length=max_length)
            logger.info("Training data prepared successfully")
            return dataset
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def setup_trainer(self, train_dataset: Any, output_dir: str, 
                     num_epochs: Optional[int] = None, batch_size: Optional[int] = None, 
                     learning_rate: Optional[float] = None, notes: str = "SimRAG Training"):
        """
        Setup trainer for fine-tuning
        
        Args:
            train_dataset: Training dataset
            output_dir: Output directory for model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            notes: Training notes for versioning
        """
        num_epochs = num_epochs or self.config.optimized_num_epochs
        batch_size = batch_size or self.config.optimized_batch_size
        learning_rate = learning_rate or self.config.learning_rate
        
        self.tuner.setup_trainer(
            train_dataset=train_dataset,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    
    def train_model(self, notes: str = "SimRAG Training") -> Optional[Any]:
        """
        Train the model
        
        Args:
            notes: Training notes for versioning
            
        Returns:
            Model version object if successful, None if training fails
            
        Raises:
            Exception: If training fails critically
        """
        try:
            logger.info(f"Starting training with notes: {notes}")
            version = self.tuner.train(notes=notes)
            if version:
                logger.info(f"Training completed successfully. Version: {version.version}")
            else:
                logger.warning("Training completed but no version object returned")
            return version
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def get_model_from_registry(self, version: Optional[str] = None) -> Optional[str]:
        """
        Get model path from registry
        
        Args:
            version: Specific version to load (default: latest)
            
        Returns:
            Model path if found
        """
        if not self.registry:
            return None
        
        try:
            if version:
                # Try to get specific version info
                model_info = getattr(self.registry, 'get_version_info', lambda x: None)(version)
            else:
                # Get active version
                model_info = getattr(self.registry, 'get_active_version', lambda: None)()
            
            if model_info:
                # Construct model path from registry info
                model_suffix = "1b" if self.config.use_laptop else "8b"
                base_dir = f"./tuned_models/llama_{model_suffix}"
                version_name = getattr(model_info, 'version', version or 'latest')
                return f"{base_dir}/{version_name}"
            
        except Exception as e:
            logger.error(f"Error loading model from registry: {e}")
        
        return None
    
    def test_performance(self, rag_system: BasicRAG, test_questions: List[str]) -> Dict[str, Any]:
        """
        Test model performance on RAG tasks
        
        Args:
            rag_system: RAG system to test
            test_questions: List of test questions
            
        Returns:
            Performance metrics dictionary with questions, answers, response_times, context_scores
            
        Raises:
            ValueError: If test_questions is empty
        """
        if not test_questions:
            raise ValueError("test_questions cannot be empty")
        
        logger.info(f"=== Testing Performance on {len(test_questions)} questions ===")
        
        results: Dict[str, Any] = {
            "questions": test_questions,
            "answers": [],
            "response_times": [],
            "context_scores": []
        }
        
        for i, question in enumerate(test_questions, 1):
            try:
                logger.info(f"Testing question {i}/{len(test_questions)}: {question[:50]}...")
                start_time = time.time()
                
                # Test with RAG system
                answer, context_docs, context_scores = rag_system.query(question)
                elapsed_time = time.time() - start_time
                
                results["answers"].append(answer)
                results["context_scores"].append(context_scores)
                results["response_times"].append(elapsed_time)
                
                avg_score = sum(context_scores) / len(context_scores) if context_scores else 0.0
                logger.info(f"  Answer length: {len(answer)} chars, Avg context score: {avg_score:.3f}, Time: {elapsed_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error testing question '{question[:50]}...': {e}")
                # Continue with other questions
                results["answers"].append("ERROR")
                results["context_scores"].append([])
                results["response_times"].append(0.0)
        
        # Calculate average metrics
        if results["context_scores"]:
            all_scores = [score for scores in results["context_scores"] if scores for score in scores]
            results["avg_context_score"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
            results["avg_response_time"] = sum(results["response_times"]) / len(results["response_times"])
        else:
            results["avg_context_score"] = 0.0
            results["avg_response_time"] = 0.0
        
        logger.info(f"Performance test completed. Avg context score: {results['avg_context_score']:.3f}")
        return results
    
    def calculate_improvement_metrics(self, baseline_results: Dict[str, Any], 
                                    improved_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvement metrics between baseline and improved results"""
        # Calculate average context scores
        baseline_scores = baseline_results.get("context_scores", [])
        improved_scores = improved_results.get("context_scores", [])
        
        if not baseline_scores or not improved_scores:
            return {
                "baseline_context_score": 0,
                "improved_context_score": 0,
                "context_improvement_percent": 0,
                "questions_tested": len(baseline_results.get("questions", []))
            }
        
        baseline_avg = sum(sum(scores) for scores in baseline_scores) / len(baseline_scores)
        improved_avg = sum(sum(scores) for scores in improved_scores) / len(improved_scores)
        
        # Calculate improvement percentage
        context_improvement = ((improved_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
        
        return {
            "baseline_context_score": baseline_avg,
            "improved_context_score": improved_avg,
            "context_improvement_percent": context_improvement,
            "questions_tested": len(baseline_results["questions"])
        }
