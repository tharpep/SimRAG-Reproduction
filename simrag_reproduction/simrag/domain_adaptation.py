"""
SimRAG Domain Adaptation
Stage 2: Fine-tune on synthetic QA pairs + Self-improvement loop
"""

import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .base import SimRAGBase
from .synthetic_qa_generation import SyntheticQAGeneration
from rag.rag_setup import BasicRAG


class DomainAdaptation(SimRAGBase):
    """SimRAG Stage 2: Domain adaptation with self-improvement loop"""
    
    def __init__(self, model_name: str = "llama3.2:1b", config=None, stage_1_model_path: Optional[str] = None):
        """
        Initialize domain adaptation trainer
        
        Args:
            model_name: Model to fine-tune
            config: TuningConfig instance
            stage_1_model_path: Path to Stage 1 model
        """
        super().__init__(model_name, config)
        self.stage_1_model_path = stage_1_model_path
        self.qa_generator = SyntheticQAGeneration(model_name, config, stage_1_model_path or None)
        
        print(f"Domain Adaptation trainer initialized")
        if stage_1_model_path:
            print(f"Using Stage 1 model: {stage_1_model_path}")
    
    def train_stage_2(self, documents: List[str], 
                     questions_per_doc: int = 2,
                     min_context_score: float = 0.7,
                     notes: str = "SimRAG Stage 2 - Domain Adaptation") -> Optional[Any]:
        """
        Train Stage 2: Domain adaptation on synthetic QA pairs
        
        Args:
            documents: List of domain documents
            questions_per_doc: Questions to generate per document
            min_context_score: Minimum context similarity threshold
            notes: Training notes for versioning
            
        Returns:
            Model version object if successful
        """
        print("=== SimRAG Stage 2: Domain Adaptation Training ===")
        
        # Generate synthetic dataset
        print("Generating synthetic QA pairs...")
        dataset = self.qa_generator.generate_synthetic_dataset(
            documents=documents,
            questions_per_doc=questions_per_doc,
            min_context_score=min_context_score
        )
        
        if not dataset["training_data"]:
            print("No training data generated")
            return None
        
        # Load model and prepare training data
        self.load_model()
        train_dataset = self.prepare_training_data(dataset["training_data"])
        
        # Setup trainer for Stage 2
        print("Setting up trainer for Stage 2...")
        output_dir = f"./tuned_models/llama_1b/stage_2"
        self.setup_trainer(
            train_dataset=train_dataset,
            output_dir=output_dir,
            num_epochs=self.config.simrag_stage_2_epochs,
            notes=notes
        )
        
        # Train Stage 2
        print("Starting Stage 2 training...")
        version = self.train_model(notes)
        
        if version:
            print(f"Stage 2 training completed!")
            print(f"Version: {version.version}")
            print(f"Training time: {version.training_time_seconds:.1f}s")
            if version.final_loss:
                print(f"Final loss: {version.final_loss:.4f}")
        
        return version
    
    def run_self_improvement_loop(self, documents: List[str], 
                                improvement_rounds: Optional[int] = None,
                                questions_per_doc: Optional[int] = None,
                                min_context_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Run self-improvement loop for Stage 3
        
        Args:
            documents: List of domain documents
            improvement_rounds: Number of improvement rounds (uses config default if None)
            questions_per_doc: Questions to generate per document (uses config default if None)
            min_context_score: Minimum context similarity threshold (uses config default if None)
            
        Returns:
            List of improvement results for each round
        """
        print("=== SimRAG Stage 3: Self-Improvement Loop ===")
        
        # Use config defaults if not provided
        improvement_rounds = improvement_rounds or self.config.simrag_improvement_rounds
        questions_per_doc = questions_per_doc or self.config.simrag_questions_per_doc
        min_context_score = min_context_score or self.config.simrag_min_context_score
        
        improvement_results = []
        current_model_path = self.stage_1_model_path
        
        for round_num in range(improvement_rounds):
            print(f"\n--- Self-Improvement Round {round_num + 1} ---")
            
            # Update QA generator to use current model
            if current_model_path:
                self.qa_generator = SyntheticQAGeneration(
                    self.model_name, 
                    self.config, 
                    current_model_path
                )
            
            # Generate new synthetic QA with current model
            print(f"Generating synthetic QA with current model...")
            dataset = self.qa_generator.generate_synthetic_dataset(
                documents=documents,
                questions_per_doc=questions_per_doc,
                min_context_score=min_context_score
            )
            
            if not dataset["training_data"]:
                print(f"Round {round_num + 1}: No training data generated, stopping")
                break
            
            # Fine-tune on new synthetic data
            print(f"Fine-tuning on {len(dataset['training_data'])} examples...")
            version = self.train_stage_2(
                documents=documents,
                questions_per_doc=questions_per_doc,
                min_context_score=min_context_score,
                notes=f"SimRAG Self-Improvement Round {round_num + 1}"
            )
            
            if version:
                # Update model path for next round
                current_model_path = self.get_model_from_registry(version.version)
                
                # Record improvement metrics
                improvement_result = {
                    "round": round_num + 1,
                    "qa_pairs_generated": dataset["dataset_info"]["total_qa_pairs"],
                    "high_quality_pairs": dataset["dataset_info"]["high_quality_pairs"],
                    "training_examples": dataset["dataset_info"]["training_examples"],
                    "quality_retention": dataset["dataset_info"]["quality_retention"],
                    "model_version": version.version,
                    "training_time": version.training_time_seconds,
                    "final_loss": version.final_loss
                }
                
                improvement_results.append(improvement_result)
                
                print(f"Round {round_num + 1} completed:")
                print(f"  Generated: {improvement_result['qa_pairs_generated']} QA pairs")
                print(f"  High quality: {improvement_result['high_quality_pairs']}")
                print(f"  Quality retention: {improvement_result['quality_retention']:.1f}%")
                print(f"  Model version: {improvement_result['model_version']}")
            else:
                print(f"Round {round_num + 1}: Training failed, stopping improvement")
                break
        
        print(f"\nSelf-improvement completed: {len(improvement_results)} rounds")
        return improvement_results
    
    def test_stage_2_performance(self, rag_system: BasicRAG, test_questions: List[str]) -> Dict[str, Any]:
        """
        Test Stage 2 model performance on RAG tasks
        
        Args:
            rag_system: RAG system to test
            test_questions: List of test questions
            
        Returns:
            Performance metrics
        """
        print("=== Testing Stage 2 Performance ===")
        return self.test_performance(rag_system, test_questions)
    
    def compare_performance(self, baseline_results: Dict[str, Any], 
                          stage_1_results: Dict[str, Any],
                          stage_2_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare performance across all stages
        
        Args:
            baseline_results: Baseline RAG performance
            stage_1_results: Stage 1 model performance
            stage_2_results: Stage 2 model performance
            
        Returns:
            Comprehensive performance comparison
        """
        print("=== Performance Comparison Across Stages ===")
        
        # Calculate improvement metrics
        stage_1_improvement = self.calculate_improvement_metrics(baseline_results, stage_1_results)
        stage_2_improvement = self.calculate_improvement_metrics(stage_1_results, stage_2_results)
        overall_improvement = self.calculate_improvement_metrics(baseline_results, stage_2_results)
        
        comparison = {
            "baseline": {
                "context_score": baseline_results.get("avg_context_score", 0),
                "questions_tested": len(baseline_results.get("questions", []))
            },
            "stage_1": {
                "context_score": stage_1_results.get("avg_context_score", 0),
                "improvement_percent": stage_1_improvement.get("context_improvement_percent", 0),
                "questions_tested": len(stage_1_results.get("questions", []))
            },
            "stage_2": {
                "context_score": stage_2_results.get("avg_context_score", 0),
                "improvement_percent": stage_2_improvement.get("context_improvement_percent", 0),
                "questions_tested": len(stage_2_results.get("questions", []))
            },
            "overall": {
                "total_improvement_percent": overall_improvement.get("context_improvement_percent", 0),
                "baseline_to_stage_2": overall_improvement
            }
        }
        
        print(f"Performance Summary:")
        print(f"  Baseline context score: {comparison['baseline']['context_score']:.3f}")
        print(f"  Stage 1 context score: {comparison['stage_1']['context_score']:.3f} (+{comparison['stage_1']['improvement_percent']:.1f}%)")
        print(f"  Stage 2 context score: {comparison['stage_2']['context_score']:.3f} (+{comparison['stage_2']['improvement_percent']:.1f}%)")
        print(f"  Overall improvement: +{comparison['overall']['total_improvement_percent']:.1f}%")
        
        return comparison


