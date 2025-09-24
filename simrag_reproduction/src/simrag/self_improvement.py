"""
Self-Improvement Module for SimRAG

Implements the self-improvement mechanisms for the SimRAG system.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from loguru import logger


class SelfImprovementModule:
    """
    Module responsible for self-improvement of the RAG system components.
    
    This includes fine-tuning the retriever and generator based on synthetic data
    and performance feedback.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the self-improvement module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Self-improvement module initialized")
    
    def fine_tune_retriever(self, retriever: SentenceTransformer, 
                           training_pairs: List[Tuple[str, str]], 
                           learning_rate: float = 1e-5) -> None:
        """
        Fine-tune the retriever using contrastive learning.
        
        Args:
            retriever: The sentence transformer retriever to fine-tune
            training_pairs: List of (query, positive_context) pairs
            learning_rate: Learning rate for fine-tuning
        """
        logger.info(f"Fine-tuning retriever with {len(training_pairs)} training pairs")
        
        # Prepare training examples
        train_examples = []
        for query, positive_context in training_pairs:
            # Create positive example
            train_examples.append(InputExample(
                texts=[query, positive_context], 
                label=1.0
            ))
            
            # Create negative examples by sampling random contexts
            negative_contexts = self._sample_negative_contexts(
                positive_context, training_pairs, num_negatives=2
            )
            for neg_context in negative_contexts:
                train_examples.append(InputExample(
                    texts=[query, neg_context], 
                    label=0.0
                ))
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(retriever)
        
        # Fine-tune the model
        retriever.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=2,
            warmup_steps=50,
            output_path="./results/simrag/retriever_checkpoints",
            show_progress_bar=True
        )
        
        logger.info("Retriever fine-tuning completed")
    
    def _sample_negative_contexts(self, positive_context: str, 
                                 training_pairs: List[Tuple[str, str]], 
                                 num_negatives: int = 2) -> List[str]:
        """
        Sample negative contexts for contrastive learning.
        
        Args:
            positive_context: The positive context to avoid
            training_pairs: All training pairs
            num_negatives: Number of negative contexts to sample
            
        Returns:
            List of negative context strings
        """
        # Get all contexts except the positive one
        all_contexts = [context for _, context in training_pairs if context != positive_context]
        
        # Sample random negative contexts
        import random
        if len(all_contexts) >= num_negatives:
            return random.sample(all_contexts, num_negatives)
        else:
            return all_contexts
    
    def evaluate_retriever_improvement(self, retriever: SentenceTransformer, 
                                     test_queries: List[str], 
                                     test_contexts: List[str]) -> Dict[str, float]:
        """
        Evaluate the improvement of the retriever.
        
        Args:
            retriever: The fine-tuned retriever
            test_queries: List of test queries
            test_contexts: List of test contexts
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating retriever improvement")
        
        # Generate embeddings
        query_embeddings = retriever.encode(test_queries)
        context_embeddings = retriever.encode(test_contexts)
        
        # Compute similarities
        similarities = []
        for i, query_emb in enumerate(query_embeddings):
            query_similarities = []
            for j, context_emb in enumerate(context_embeddings):
                # Compute cosine similarity
                similarity = torch.cosine_similarity(
                    torch.tensor(query_emb).unsqueeze(0),
                    torch.tensor(context_emb).unsqueeze(0)
                ).item()
                query_similarities.append((similarity, j))
            
            # Sort by similarity
            query_similarities.sort(reverse=True)
            similarities.append(query_similarities)
        
        # Compute metrics
        metrics = {
            "avg_top1_similarity": 0.0,
            "avg_top5_similarity": 0.0,
            "avg_top10_similarity": 0.0
        }
        
        for query_sims in similarities:
            if len(query_sims) > 0:
                metrics["avg_top1_similarity"] += query_sims[0][0]
            if len(query_sims) > 4:
                metrics["avg_top5_similarity"] += sum(sim[0] for sim in query_sims[:5]) / 5
            if len(query_sims) > 9:
                metrics["avg_top10_similarity"] += sum(sim[0] for sim in query_sims[:10]) / 10
        
        # Average over all queries
        num_queries = len(test_queries)
        for key in metrics:
            metrics[key] /= num_queries
        
        logger.info(f"Retriever evaluation completed: {metrics}")
        return metrics
    
    def compute_improvement_score(self, baseline_metrics: Dict[str, float], 
                                 improved_metrics: Dict[str, float]) -> float:
        """
        Compute an overall improvement score.
        
        Args:
            baseline_metrics: Metrics from baseline system
            improved_metrics: Metrics from improved system
            
        Returns:
            Overall improvement score
        """
        # Weight different metrics
        weights = {
            "f1": 0.3,
            "em": 0.2,
            "recall_at_5": 0.2,
            "recall_at_10": 0.2,
            "ndcg_at_10": 0.1
        }
        
        total_improvement = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in baseline_metrics and metric in improved_metrics:
                improvement = improved_metrics[metric] - baseline_metrics[metric]
                total_improvement += improvement * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_improvement / total_weight
        else:
            return 0.0
