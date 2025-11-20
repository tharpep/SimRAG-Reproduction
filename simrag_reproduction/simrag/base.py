"""
SimRAG Base Class
Common functionality for all SimRAG stages
"""

import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..logging_config import get_logger
from ..tuning.basic_tuning import BasicTuner
from ..tuning.model_registry import get_model_registry
from ..tuning.ollama_integration import register_model_with_ollama
from ..config import get_tuning_config
from ..rag.rag_setup import BasicRAG

logger = get_logger(__name__)


class SimRAGBase:
    """Base class for all SimRAG stages with common functionality"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", config: Optional[Any] = None):
        """
        Initialize SimRAG base functionality
        
        Args:
            model_name: Model to fine-tune
            config: TuningConfig instance (optional)
        """
        self.model_name = model_name
        self.experiment_run_id = None  # Will be set before training to link versions
        self.ollama_model_name = None  # Will be set after Ollama registration
        try:
            self.config = config or get_tuning_config()
            self.tuner = BasicTuner(model_name, device=self.config.device, config=self.config)
            self.registry = get_model_registry(self.config)
            logger.info(f"SimRAG base initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SimRAG base: {e}")
            raise
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the model for training
        
        Args:
            model_path: Optional path to fine-tuned model to load from.
                       If None, loads from base model (HuggingFace).
        
        Raises:
            Exception: If model loading fails
        """
        try:
            if model_path:
                logger.info(f"Loading fine-tuned model from: {model_path}")
            else:
                logger.info("Loading base model...")
            
            # Use BasicTuner's load_model which now supports model_path parameter
            self.tuner.load_model(model_path=model_path)
            
            if model_path:
                logger.info(f"Fine-tuned model loaded successfully from {model_path}")
            else:
                logger.info("Base model loaded successfully")
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
            
            # Train the model - tuner.train() now creates version BEFORE training
            # and sets output_dir correctly, so trainer saves directly to version directory
            # Pass experiment_run_id so it's set when version is created (not after)
            version = self.tuner.train(notes=notes, experiment_run_id=self.experiment_run_id)
            
            if version:
                logger.info(f"Training completed successfully. Version: {version.version}")
                logger.info(f"Training time: {version.training_time_seconds:.1f}s")
                if version.final_loss:
                    logger.info(f"Final loss: {version.final_loss:.4f}")
                
                if self.experiment_run_id:
                    logger.info(f"Version {version.version} linked to experiment run: {self.experiment_run_id}")
                
                # Ensure model is saved (trainer already saved during training, but ensure tokenizer is saved)
                # The trainer already saved to the correct version directory
                version_output_dir = self.tuner.trainer.args.output_dir
                self.tuner.save_model(version_output_dir)
                logger.info(f"Model saved to: {version_output_dir}")
                
                # Register with Ollama for fast inference
                self._register_with_ollama(version_output_dir, version.version)
            else:
                logger.warning("Training completed but no version object returned")
            
            return version
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _register_with_ollama(self, model_path: str, version: str):
        """
        Register trained model with Ollama for fast inference
        
        Args:
            model_path: Path to model directory with adapters
            version: Model version string
        """
        try:
            # Determine stage from model path
            if "stage_1" in model_path:
                stage = "stage_1"
            elif "stage_2" in model_path:
                stage = "stage_2"
            else:
                stage = "unknown"
            
            logger.info(f"Registering model with Ollama (stage={stage}, version={version})...")
            
            ollama_model_name = register_model_with_ollama(
                adapter_path=model_path,
                stage=stage,
                version=version,
                model_size=self.config.model_size
            )
            
            if ollama_model_name:
                logger.info(f"✓ Model registered with Ollama: {ollama_model_name}")
                logger.info(f"  You can test it with: ollama run {ollama_model_name}")
                # Store the Ollama model name for later use
                self.ollama_model_name = ollama_model_name
            else:
                logger.warning("Could not register model with Ollama (Ollama may not be installed)")
                self.ollama_model_name = None
                
        except Exception as e:
            logger.warning(f"Failed to register with Ollama: {e}")
            self.ollama_model_name = None
    
    def get_model_from_registry(self, version: Optional[str] = None, stage: Optional[str] = None) -> Optional[str]:
        """
        Get model path from registry
        
        Args:
            version: Specific version to load (default: latest)
            stage: Stage name ("stage_1" or "stage_2") - if None, searches both
            
        Returns:
            Model path if found (absolute path)
        """
        if not self.registry:
            return None
        
        try:
            if version:
                # Get specific version
                model_info = self.registry.get_version(version)
            else:
                # Get active version
                model_info = self.registry.get_active_version()
            
            if model_info:
                # Construct model path from registry info
                model_suffix = "1b" if self.config.model_size == "small" else "8b"
                base_dir = f"./tuned_models/model_{model_suffix}"
                version_name = model_info.version
                
                # Try to find the model in stage directories
                if stage:
                    # Check specific stage
                    model_path = f"{base_dir}/{stage}/{version_name}"
                else:
                    # Search both stages (try stage_2 first, then stage_1)
                    model_path = None
                    for stage_name in ["stage_2", "stage_1"]:
                        candidate_path = f"{base_dir}/{stage_name}/{version_name}"
                        from pathlib import Path
                        if Path(candidate_path).exists():
                            model_path = candidate_path
                            break
                    
                    if not model_path:
                        # Fallback: try without stage (old format)
                        model_path = f"{base_dir}/{version_name}"
                
                # Convert to absolute path
                from pathlib import Path
                abs_path = Path(model_path).resolve()
                if abs_path.exists():
                    logger.info(f"Found model at: {abs_path}")
                    return str(abs_path)
                else:
                    logger.warning(f"Model path does not exist: {abs_path}")
                    return None
            
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
        
        # Import evaluate_answer_quality function
        from ..experiments.utils import evaluate_answer_quality
        
        logger.info(f"=== Testing Performance on {len(test_questions)} questions ===")
        
        results: Dict[str, Any] = {
            "questions": test_questions,
            "answers": [],
            "response_times": [],
            "context_scores": [],
            "answer_quality_scores": []  # NEW: Answer quality metrics
        }
        
        for i, question in enumerate(test_questions, 1):
            try:
                logger.info(f"Testing question {i}/{len(test_questions)}: {question[:50]}...")
                start_time = time.time()
                
                # Test with RAG system
                answer, context_docs, context_scores = rag_system.query(question)
                elapsed_time = time.time() - start_time
                
                # Evaluate answer quality
                context_text = "\n\n".join(context_docs)
                quality_scores = evaluate_answer_quality(question, answer, context_text)
                
                results["answers"].append(answer)
                results["context_scores"].append(context_scores)
                results["response_times"].append(elapsed_time)
                results["answer_quality_scores"].append(quality_scores)
                
                avg_score = sum(context_scores) / len(context_scores) if context_scores else 0.0
                logger.info(f"  Answer length: {len(answer)} chars, Context: {avg_score:.3f}, Quality: {quality_scores['overall_score']:.3f}, Time: {elapsed_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error testing question '{question[:50]}...': {e}")
                # Continue with other questions
                results["answers"].append("ERROR")
                results["context_scores"].append([])
                results["response_times"].append(0.0)
                results["answer_quality_scores"].append({
                    "length_score": 0.0,
                    "context_relevance": 0.0,
                    "not_refusal": 0.0,
                    "question_relevance": 0.0,
                    "overall_score": 0.0
                })
        
        # Calculate average metrics
        if results["context_scores"]:
            all_scores = [score for scores in results["context_scores"] if scores for score in scores]
            results["avg_context_score"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
            results["avg_response_time"] = sum(results["response_times"]) / len(results["response_times"])
        else:
            results["avg_context_score"] = 0.0
            results["avg_response_time"] = 0.0
        
        # Calculate average quality scores
        if results["answer_quality_scores"]:
            metric_keys = ["length_score", "context_relevance", "not_refusal", "question_relevance", "overall_score"]
            for key in metric_keys:
                values = [q[key] for q in results["answer_quality_scores"] if isinstance(q, dict)]
                results[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
        
        logger.info(f"Performance test completed. Avg context: {results['avg_context_score']:.3f}, Avg quality: {results.get('avg_overall_score', 0.0):.3f}")
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
