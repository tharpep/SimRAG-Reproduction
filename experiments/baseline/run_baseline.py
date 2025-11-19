"""
Run Baseline RAG Experiment
Tests vanilla RAG system (no fine-tuning) on domain documents
"""

import json
import time
from pathlib import Path
from datetime import datetime

from rag.rag_setup import BasicRAG
from config import get_rag_config
from experiments.utils import load_documents_from_folder, get_test_questions
from logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def run_baseline_test(
    documents_folder: str = "../HTML_DOCS",
    test_questions: list = None,
    output_file: str = None,
    use_timestamp: bool = True
) -> dict:
    """
    Run baseline RAG test
    
    Args:
        documents_folder: Path to folder containing documents
        test_questions: List of test questions (uses default if None)
        output_file: Path to save results JSON (optional)
        
    Returns:
        Dictionary with baseline test results
    """
    logger.info("=== Baseline RAG Experiment ===")
    
    # Get test questions
    if test_questions is None:
        test_questions = get_test_questions()
    
    # Initialize RAG system - use local model (Ollama) for testing
    logger.info("Initializing RAG system...")
    rag_config = get_rag_config()
    rag = BasicRAG(
        collection_name="baseline_experiment",
        use_persistent=False,  # Use in-memory for experiments
        force_provider="ollama"  # Use local model for testing
    )
    
    # Load documents
    logger.info(f"Loading documents from {documents_folder}...")
    documents = load_documents_from_folder(documents_folder, include_html=True)
    
    if not documents:
        raise ValueError(f"No documents found in {documents_folder}")
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Ingest documents
    logger.info("Ingesting documents into vector store...")
    num_added = rag.add_documents(documents)
    logger.info(f"Indexed {num_added} document chunks")
    
    # Get collection stats
    stats = rag.get_stats()
    logger.info(f"Collection stats: {stats.get('document_count', 0)} points")
    
    # Run test questions
    logger.info(f"Testing on {len(test_questions)} questions...")
    results = {
        "experiment_type": "baseline",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": rag_config.model_name,
            "use_ollama": rag_config.use_ollama,
            "top_k": rag_config.top_k,
            "similarity_threshold": rag_config.similarity_threshold
        },
        "dataset": {
            "documents_folder": documents_folder,
            "num_documents": len(documents),
            "num_chunks_indexed": num_added
        },
        "questions": test_questions,
        "answers": [],
        "context_scores": [],
        "response_times": [],
        "context_docs": []
    }
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"Question {i}/{len(test_questions)}: {question[:60]}...")
        
        start_time = time.time()
        try:
            answer, context_docs, context_scores = rag.query(question)
            elapsed = time.time() - start_time
            
            results["answers"].append(answer)
            results["context_scores"].append(context_scores)
            results["response_times"].append(elapsed)
            results["context_docs"].append(context_docs)
            
            avg_score = sum(context_scores) / len(context_scores) if context_scores else 0.0
            logger.info(f"  Answer length: {len(answer)} chars, Avg score: {avg_score:.3f}, Time: {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"  Error processing question: {e}")
            results["answers"].append("ERROR")
            results["context_scores"].append([])
            results["response_times"].append(0.0)
            results["context_docs"].append([])
    
    # Calculate summary metrics
    all_scores = [score for scores in results["context_scores"] if scores for score in scores]
    results["summary"] = {
        "avg_context_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "avg_response_time": sum(results["response_times"]) / len(results["response_times"]),
        "total_questions": len(test_questions),
        "successful_queries": len([a for a in results["answers"] if a != "ERROR"])
    }
    
    logger.info(f"Baseline experiment completed!")
    logger.info(f"  Avg context score: {results['summary']['avg_context_score']:.3f}")
    logger.info(f"  Avg response time: {results['summary']['avg_response_time']:.2f}s")
    
    # Save results
    if output_file:
        # Add timestamp if not already present and use_timestamp is True
        if use_timestamp:
            from experiments.utils import get_timestamped_filename, has_timestamp
            if not has_timestamp(output_file):
                base_name = Path(output_file).stem
                output_file = get_timestamped_filename(base_name, "json")
        
        output_path = Path(__file__).parent / "results" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
        
        # Store the actual filename used
        results["_saved_filename"] = str(output_path)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline RAG experiment")
    parser.add_argument("--documents", type=str, default="../HTML_DOCS",
                       help="Path to documents folder (default: ../HTML_DOCS)")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                       help="Output filename (default: baseline_results.json, auto-timestamped)")
    parser.add_argument("--no-timestamp", action="store_true",
                       help="Don't add timestamp to filename (may overwrite existing)")
    
    args = parser.parse_args()
    
    try:
        results = run_baseline_test(
            documents_folder=args.documents,
            output_file=args.output,
            use_timestamp=not args.no_timestamp
        )
        print("\n=== Baseline Experiment Complete ===")
        saved_file = results.get("_saved_filename", args.output)
        print(f"Results saved to: {saved_file}")
    except Exception as e:
        logger.error(f"Baseline experiment failed: {e}")
        sys.exit(1)

