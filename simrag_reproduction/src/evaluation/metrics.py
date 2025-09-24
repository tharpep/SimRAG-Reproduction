"""
Evaluation Metrics for RAG Systems

Implements various metrics for evaluating RAG system performance including
EM, F1, Recall@k, and nDCG as specified in the project requirements.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
import torch
from loguru import logger


def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation by removing extra whitespace and converting to lowercase.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text


def exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match score.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        Exact match score (0.0 to 1.0)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    matches = 0
    for pred, ref in zip(predictions, references):
        if normalize_text(pred) == normalize_text(ref):
            matches += 1
    
    return matches / len(predictions)


def f1_score_qa(predictions: List[str], references: List[str]) -> float:
    """
    Compute F1 score for QA evaluation.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    total_f1 = 0.0
    
    for pred, ref in zip(predictions, references):
        # Tokenize and normalize
        pred_tokens = set(normalize_text(pred).split())
        ref_tokens = set(normalize_text(ref).split())
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            f1 = 1.0
        elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1 = 0.0
        else:
            # Compute precision and recall
            common_tokens = pred_tokens.intersection(ref_tokens)
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(ref_tokens)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        
        total_f1 += f1
    
    return total_f1 / len(predictions)


def recall_at_k(retrieved_docs: List[List[Dict[str, Any]]], 
                relevant_docs: List[List[int]], 
                k: int = 10) -> float:
    """
    Compute Recall@k for retrieval evaluation.
    
    Args:
        retrieved_docs: List of retrieved documents for each query
        relevant_docs: List of relevant document indices for each query
        k: Number of top documents to consider
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if len(retrieved_docs) != len(relevant_docs):
        raise ValueError("Retrieved docs and relevant docs must have the same length")
    
    total_recall = 0.0
    
    for ret_docs, rel_docs in zip(retrieved_docs, relevant_docs):
        # Get top-k retrieved document indices
        top_k_indices = [doc["index"] for doc in ret_docs[:k]]
        
        # Compute recall
        if len(rel_docs) == 0:
            recall = 1.0 if len(top_k_indices) == 0 else 0.0
        else:
            relevant_retrieved = len(set(top_k_indices).intersection(set(rel_docs)))
            recall = relevant_retrieved / len(rel_docs)
        
        total_recall += recall
    
    return total_recall / len(retrieved_docs)


def ndcg_at_k(retrieved_docs: List[List[Dict[str, Any]]], 
              relevant_docs: List[List[int]], 
              k: int = 10) -> float:
    """
    Compute nDCG@k for retrieval evaluation.
    
    Args:
        retrieved_docs: List of retrieved documents for each query
        relevant_docs: List of relevant document indices for each query
        k: Number of top documents to consider
        
    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    if len(retrieved_docs) != len(relevant_docs):
        raise ValueError("Retrieved docs and relevant docs must have the same length")
    
    total_ndcg = 0.0
    
    for ret_docs, rel_docs in zip(retrieved_docs, relevant_docs):
        # Get top-k retrieved documents
        top_k_docs = ret_docs[:k]
        
        # Compute DCG
        dcg = 0.0
        for i, doc in enumerate(top_k_docs):
            if doc["index"] in rel_docs:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Compute IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(rel_docs), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        # Compute nDCG
        if idcg == 0:
            ndcg = 1.0 if dcg == 0 else 0.0
        else:
            ndcg = dcg / idcg
        
        total_ndcg += ndcg
    
    return total_ndcg / len(retrieved_docs)


def semantic_similarity(predictions: List[str], references: List[str]) -> float:
    """
    Compute semantic similarity using sentence transformers.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        Average semantic similarity score (0.0 to 1.0)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    # Use a lightweight sentence transformer for efficiency
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute embeddings
    pred_embeddings = model.encode(predictions)
    ref_embeddings = model.encode(references)
    
    # Compute cosine similarities
    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        similarity = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
        similarities.append(similarity)
    
    return np.mean(similarities)


def compute_metrics(predictions: List[str], references: List[str], 
                   retrieved_docs: Optional[List[List[Dict[str, Any]]]] = None,
                   relevant_docs: Optional[List[List[int]]] = None,
                   k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for RAG systems.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        retrieved_docs: List of retrieved documents for each query (optional)
        relevant_docs: List of relevant document indices for each query (optional)
        k_values: List of k values for Recall@k and nDCG@k
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Computing evaluation metrics")
    
    metrics = {}
    
    # Answer quality metrics
    metrics["em"] = exact_match(predictions, references)
    metrics["f1"] = f1_score_qa(predictions, references)
    metrics["semantic_similarity"] = semantic_similarity(predictions, references)
    
    # Retrieval metrics (if provided)
    if retrieved_docs is not None and relevant_docs is not None:
        for k in k_values:
            metrics[f"recall_at_{k}"] = recall_at_k(retrieved_docs, relevant_docs, k)
            metrics[f"ndcg_at_{k}"] = ndcg_at_k(retrieved_docs, relevant_docs, k)
    
    logger.info(f"Computed metrics: {list(metrics.keys())}")
    return metrics


def compute_improvement_metrics(baseline_metrics: Dict[str, float], 
                               improved_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compute improvement metrics comparing baseline and improved systems.
    
    Args:
        baseline_metrics: Metrics from baseline system
        improved_metrics: Metrics from improved system
        
    Returns:
        Dictionary of improvement metrics (absolute and relative)
    """
    improvements = {}
    
    for metric in baseline_metrics.keys():
        if metric in improved_metrics:
            baseline_val = baseline_metrics[metric]
            improved_val = improved_metrics[metric]
            
            # Absolute improvement
            improvements[f"{metric}_improvement"] = improved_val - baseline_val
            
            # Relative improvement (percentage)
            if baseline_val != 0:
                improvements[f"{metric}_relative_improvement"] = (improved_val - baseline_val) / baseline_val * 100
            else:
                improvements[f"{metric}_relative_improvement"] = 0.0
    
    return improvements


def format_metrics_report(metrics: Dict[str, float], title: str = "Evaluation Results") -> str:
    """
    Format metrics into a readable report.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the report
        
    Returns:
        Formatted report string
    """
    report = f"\n{title}\n"
    report += "=" * len(title) + "\n\n"
    
    # Group metrics by category
    answer_metrics = {k: v for k, v in metrics.items() if k in ["em", "f1", "semantic_similarity"]}
    retrieval_metrics = {k: v for k, v in metrics.items() if k.startswith(("recall_at_", "ndcg_at_"))}
    improvement_metrics = {k: v for k, v in metrics.items() if k.endswith(("_improvement", "_relative_improvement"))}
    
    if answer_metrics:
        report += "Answer Quality Metrics:\n"
        for metric, value in answer_metrics.items():
            report += f"  {metric.upper()}: {value:.4f}\n"
        report += "\n"
    
    if retrieval_metrics:
        report += "Retrieval Metrics:\n"
        for metric, value in retrieval_metrics.items():
            report += f"  {metric.upper()}: {value:.4f}\n"
        report += "\n"
    
    if improvement_metrics:
        report += "Improvement Metrics:\n"
        for metric, value in improvement_metrics.items():
            if "relative" in metric:
                report += f"  {metric.upper()}: {value:.2f}%\n"
            else:
                report += f"  {metric.upper()}: {value:.4f}\n"
        report += "\n"
    
    return report
