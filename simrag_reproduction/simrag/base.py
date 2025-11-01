"""
SimRAG Base Class
Common functionality for all SimRAG stages
"""

import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tuning.basic_tuning import BasicTuner
from tuning.model_registry import get_model_registry
from config import get_tuning_config
from rag.rag_setup import BasicRAG


class SimRAGBase:
    """Base class for all SimRAG stages with common functionality"""
    
    def __init__(self, model_name: str = "llama3.2:1b", config=None):
        """
        Initialize SimRAG base functionality
        
        Args:
            model_name: Model to fine-tune
            config: TuningConfig instance
        """
        self.model_name = model_name
        self.config = config or get_tuning_config()
        self.tuner = BasicTuner(model_name, config=self.config)
        self.registry = get_model_registry(self.config)
        
        print(f"SimRAG base initialized with model: {model_name}")
    
    def load_model(self):
        """Load the model for training"""
        print("Loading model...")
        self.tuner.load_model()
    
    def prepare_training_data(self, data: List[str], max_length: Optional[int] = None) -> Any:
        """
        Prepare training data
        
        Args:
            data: List of training examples
            max_length: Maximum sequence length
            
        Returns:
            Prepared training dataset
        """
        max_length = max_length or self.config.max_length
        return self.tuner.prepare_data(data, max_length=max_length)
    
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
            Model version object if successful
        """
        print("Starting training...")
        return self.tuner.train(notes=notes)
    
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
            print(f"Error loading model from registry: {e}")
        
        return None
    
    def test_performance(self, rag_system: BasicRAG, test_questions: List[str]) -> Dict[str, Any]:
        """
        Test model performance on RAG tasks
        
        Args:
            rag_system: RAG system to test
            test_questions: List of test questions
            
        Returns:
            Performance metrics
        """
        print("=== Testing Performance ===")
        
        results = {
            "questions": test_questions,
            "answers": [],
            "response_times": [],
            "context_scores": []
        }
        
        for question in test_questions:
            print(f"Testing: {question}")
            
            # Test with RAG system
            answer, context_docs, context_scores = rag_system.query(question)
            
            results["answers"].append(answer)
            results["context_scores"].append(context_scores)
            
            print(f"Answer: {answer[:100]}...")
            print(f"Context scores: {context_scores}")
            print()
        
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
