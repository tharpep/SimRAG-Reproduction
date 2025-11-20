"""
SimRAG Domain Adaptation
Stage 2: Fine-tune on synthetic QA pairs with integrated self-improvement loop

Stage 2 now includes the self-improvement loop by default, controlled by
simrag_improvement_rounds config. Each round generates new synthetic QA pairs
using the improved model from the previous round, enabling iterative refinement.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..logging_config import get_logger
from .base import SimRAGBase
from .synthetic_qa_generation import SyntheticQAGeneration
from ..rag.rag_setup import BasicRAG

logger = get_logger(__name__)


class DomainAdaptation(SimRAGBase):
    """SimRAG Stage 2: Domain adaptation with integrated self-improvement loop"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", config: Optional[Any] = None, stage_1_model_path: Optional[str] = None):
        """
        Initialize domain adaptation trainer
        
        Args:
            model_name: Model to fine-tune
            config: TuningConfig instance (optional)
            stage_1_model_path: Path to Stage 1 model (optional)
        """
        super().__init__(model_name, config)
        self.stage_1_model_path = stage_1_model_path
        try:
            self.qa_generator = SyntheticQAGeneration(model_name, config, stage_1_model_path or None)
            logger.info("Domain Adaptation trainer initialized")
            if stage_1_model_path:
                logger.info(f"Using Stage 1 model: {stage_1_model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Domain Adaptation trainer: {e}")
            raise
    
    def _train_single_round(self, documents: List[str],
                           current_model_path: Optional[str],
                           round_num: int,
                           questions_per_doc: int,
                           min_context_score: float,
                           notes: str) -> Optional[Any]:
        """
        Train a single round of Stage 2 (helper method)
        
        Args:
            documents: List of domain documents
            current_model_path: Path to current model (Stage 1 or previous round)
            round_num: Current round number
            questions_per_doc: Questions to generate per document
            min_context_score: Minimum context similarity threshold
            notes: Training notes for versioning
            
        Returns:
            Model version object if successful
        """
        logger.info(f"--- Stage 2 Round {round_num} ---")
        
        # Generate synthetic dataset with current model
        logger.info("Generating synthetic QA pairs...")
        try:
            dataset = self.qa_generator.generate_synthetic_dataset(
                documents=documents,
                questions_per_doc=questions_per_doc,
                min_context_score=min_context_score
            )
        except Exception as e:
            logger.error(f"Round {round_num}: Failed to generate synthetic dataset: {e}")
            raise
        
        if not dataset.get("training_data"):
            logger.warning(f"Round {round_num}: No training data generated")
            return None
        
        logger.info(f"Generated {len(dataset['training_data'])} training examples")
        
        # Load model - use provided model path or base model
        if current_model_path:
            logger.info(f"Loading model for Round {round_num}: {current_model_path}")
            self.load_model(model_path=current_model_path)
        else:
            logger.warning(f"⚠️  Round {round_num}: No model path provided - using BASE model")
            logger.warning("   This may not follow SimRAG methodology. Stage 2 should continue from Stage 1.")
            logger.info("Loading base model (fallback mode)")
            self.load_model()
        
        train_dataset = self.prepare_training_data(dataset["training_data"])
        
        # Setup trainer
        logger.info(f"Setting up trainer for Round {round_num}...")
        output_dir = self.config.get_stage_output_dir("stage_2")
        try:
            self.setup_trainer(
                train_dataset=train_dataset,
                output_dir=output_dir,
                num_epochs=self.config.simrag_stage_2_epochs,
                notes=notes
            )
        except Exception as e:
            logger.error(f"Round {round_num}: Failed to setup trainer: {e}")
            raise
        
        # Train
        logger.info(f"Starting training for Round {round_num}...")
        try:
            version = self.train_model(notes)
            
            if version:
                logger.info(f"Round {round_num} completed! Version: {version.version}")
                logger.info(f"Training time: {version.training_time_seconds:.1f}s")
                if version.final_loss:
                    logger.info(f"Final loss: {version.final_loss:.4f}")
            else:
                logger.warning(f"Round {round_num} completed but no version returned")
            
            return version
        except Exception as e:
            logger.error(f"Round {round_num} training failed: {e}")
            raise
    
    def train_stage_2(self, documents: List[str], 
                     questions_per_doc: Optional[int] = None,
                     min_context_score: Optional[float] = None,
                     improvement_rounds: Optional[int] = None,
                     notes: str = "SimRAG Stage 2 - Domain Adaptation") -> Optional[Any]:
        """
        Train Stage 2: Domain adaptation with integrated self-improvement loop
        
        Runs multiple rounds of training based on simrag_improvement_rounds config.
        Each round generates new synthetic QA pairs using the improved model from the previous round.
        
        Args:
            documents: List of domain documents
            questions_per_doc: Questions to generate per document (uses config default if None)
            min_context_score: Minimum context similarity threshold (uses config default if None)
            improvement_rounds: Number of improvement rounds (uses config default if None)
            notes: Training notes for versioning
            
        Returns:
            Final model version object if successful
        """
        logger.info("=== SimRAG Stage 2: Domain Adaptation Training ===")
        
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        # Use config defaults if not provided
        improvement_rounds = improvement_rounds or self.config.simrag_improvement_rounds
        questions_per_doc = questions_per_doc or self.config.simrag_questions_per_doc
        min_context_score = min_context_score or self.config.simrag_min_context_score
        
        if improvement_rounds < 1:
            raise ValueError("improvement_rounds must be at least 1")
        
        logger.info(f"Stage 2 will run {improvement_rounds} round(s)")
        if improvement_rounds > 1:
            logger.info("Self-improvement loop enabled: each round uses the improved model from the previous round")
        
        current_model_path = self.stage_1_model_path
        final_version = None
        
        # Run improvement rounds
        for round_num in range(1, improvement_rounds + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Stage 2 Round {round_num}/{improvement_rounds}")
            logger.info(f"{'='*60}")
            
            # Update QA generator to use current model (for rounds > 1)
            if round_num > 1 and current_model_path:
                logger.info(f"Updating QA generator to use improved model from Round {round_num - 1}")
                self.qa_generator = SyntheticQAGeneration(
                    self.model_name,
                    self.config,
                    current_model_path
                )
            
            # Train single round
            try:
                round_notes = f"{notes} - Round {round_num}/{improvement_rounds}"
                version = self._train_single_round(
                    documents=documents,
                    current_model_path=current_model_path,
                    round_num=round_num,
                    questions_per_doc=questions_per_doc,
                    min_context_score=min_context_score,
                    notes=round_notes
                )
                
                if version:
                    final_version = version
                    # Get model path for next round
                    current_model_path = self.get_model_from_registry(version.version, stage="stage_2")
                    logger.info(f"✓ Round {round_num} successful - model saved as version {version.version}")
                else:
                    logger.warning(f"Round {round_num} did not produce a model version - stopping")
                    break
                    
            except Exception as e:
                logger.error(f"Round {round_num} failed: {e}")
                if round_num == 1:
                    # First round failed - propagate error
                    raise
                else:
                    # Later round failed - return last successful version
                    logger.warning(f"Stopping after {round_num - 1} successful rounds")
                    break
        
        if final_version:
            logger.info(f"\n{'='*60}")
            logger.info(f"Stage 2 Training Complete!")
            logger.info(f"Total rounds completed: {round_num}")
            logger.info(f"Final model version: {final_version.version}")
            logger.info(f"{'='*60}")
        else:
            logger.error("Stage 2 training failed - no model version produced")
        
        return final_version
    
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
        logger.info("=== Performance Comparison Across Stages ===")
        
        # Calculate improvement metrics
        try:
            stage_1_improvement = self.calculate_improvement_metrics(baseline_results, stage_1_results)
            stage_2_improvement = self.calculate_improvement_metrics(stage_1_results, stage_2_results)
            overall_improvement = self.calculate_improvement_metrics(baseline_results, stage_2_results)
        except Exception as e:
            logger.error(f"Failed to calculate improvement metrics: {e}")
            raise
        
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
        
        logger.info(f"Performance Summary:")
        logger.info(f"  Baseline context score: {comparison['baseline']['context_score']:.3f}")
        logger.info(f"  Stage 1 context score: {comparison['stage_1']['context_score']:.3f} (+{comparison['stage_1']['improvement_percent']:.1f}%)")
        logger.info(f"  Stage 2 context score: {comparison['stage_2']['context_score']:.3f} (+{comparison['stage_2']['improvement_percent']:.1f}%)")
        logger.info(f"  Overall improvement: +{comparison['overall']['total_improvement_percent']:.1f}%")
        
        return comparison


