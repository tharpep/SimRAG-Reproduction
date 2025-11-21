"""
Test Model Locally with Transformers + 4-bit Quantization
Alternative to Ollama for Windows - uses transformers directly with BitsAndBytes quantization
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import gc

from ..config import get_rag_config, get_tuning_config
from ..logging_config import setup_logging, get_logger
from .utils import (
    load_documents_from_folder,
    get_test_questions,
    evaluate_answer_quality,
    get_system_metadata,
    get_timestamped_filename,
    has_timestamp
)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import chromadb
    from chromadb.utils import embedding_functions
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Setup logging
setup_logging()
logger = get_logger(__name__)


def test_model_local(
    model_path: str,
    stage: str,
    documents_folder: str = "../../data/documents",
    test_questions: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    use_timestamp: bool = True
) -> Dict[str, Any]:
    """
    Test model locally using transformers + 4-bit quantization

    This is an alternative to Ollama that works reliably on Windows.
    Uses 4-bit quantization to fit in VRAM (uses ~2-3GB for 1.5B model).

    Args:
        model_path: Path to LoRA adapter directory
        stage: Stage name ("stage_1" or "stage_2")
        documents_folder: Path to documents folder
        test_questions: List of test questions (uses default if None)
        output_file: Path to save results JSON (optional)
        use_timestamp: Add timestamp to output filename

    Returns:
        Dictionary with test results
    """

    if not DEPS_AVAILABLE:
        raise ImportError(f"Required dependencies not available: {IMPORT_ERROR}")

    logger.info(f"=== Testing Model Locally: {model_path} ===")
    logger.info("Using transformers + 4-bit quantization (BitsAndBytes)")

    # Get test questions
    if test_questions is None:
        test_questions = get_test_questions()

    # Resolve paths
    model_path_abs = Path(model_path).resolve()

    # Verify adapter files exist
    adapter_file = model_path_abs / "adapter_model.safetensors"
    adapter_config_file = model_path_abs / "adapter_config.json"

    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter file not found: {adapter_file}")
    if not adapter_config_file.exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_config_file}")

    logger.info(f"✓ Found adapter files:")
    logger.info(f"  - {adapter_file.name} ({adapter_file.stat().st_size / (1024**2):.1f} MB)")
    logger.info(f"  - {adapter_config_file.name}")

    # Load adapter config to get base model
    with open(adapter_config_file, 'r') as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    logger.info(f"✓ Base model: {base_model_name}")
    logger.info(f"  LoRA r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}")

    # Extract version and checkpoint from path
    model_version = model_path_abs.parent.name if "checkpoint" in model_path_abs.name else model_path_abs.name
    checkpoint_name = model_path_abs.name if "checkpoint" in model_path_abs.name else None

    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available - will run on CPU (VERY SLOW!)")
        logger.warning("  This is not recommended for testing.")
        use_quantization = False
    else:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"✓ GPU: {gpu_name}")
        logger.info(f"  Total VRAM: {total_vram:.1f} GB")
        use_quantization = True

    # Load model with 4-bit quantization
    logger.info("\n=== Loading Model ===")
    logger.info("This will take 1-3 minutes...")

    if use_quantization:
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        logger.info("Using 4-bit quantization (nf4)")
    else:
        bnb_config = None
        logger.info("Using FP32 (CPU mode)")

    start_load = time.time()

    try:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if use_quantization else "cpu",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_quantization else torch.float32
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load LoRA adapters
        model = PeftModel.from_pretrained(model, str(model_path_abs))
        model.eval()

        elapsed_load = time.time() - start_load

        if use_quantization:
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"✓ Model loaded in {elapsed_load:.1f}s")
            logger.info(f"  VRAM used: {vram_used:.2f} GB")
        else:
            logger.info(f"✓ Model loaded in {elapsed_load:.1f}s (CPU mode)")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Load documents
    logger.info(f"\n=== Loading Documents ===")
    logger.info(f"From: {documents_folder}")
    documents = load_documents_from_folder(documents_folder, include_html=True)

    if not documents:
        raise ValueError(f"No documents found in {documents_folder}")

    logger.info(f"✓ Loaded {len(documents)} documents")

    # Setup ChromaDB for retrieval (in-memory)
    logger.info("\n=== Setting up RAG ===")
    client = chromadb.Client()

    # CRITICAL: Use same embedding model as baseline
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection = client.create_collection(
        name="model_test_local",
        embedding_function=embedding_fn
    )

    # Add documents
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(documents=documents, ids=doc_ids)
    logger.info(f"✓ Indexed {len(documents)} documents")
    logger.info(f"  Embedding model: sentence-transformers/all-MiniLM-L6-v2")

    # RAG config (match baseline exactly)
    TOP_K = 5
    TEMPERATURE = 0.7
    MAX_TOKENS = 512

    logger.info(f"  top_k={TOP_K}, temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}")

    def query_rag(question: str):
        """Query RAG system (matches baseline implementation)"""
        # Retrieve context
        results = collection.query(
            query_texts=[question],
            n_results=TOP_K
        )

        context_docs = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []

        # Convert distances to similarity scores (1 - distance)
        context_scores = [max(0.0, 1.0 - d) for d in distances]

        context_text = "\n\n".join(context_docs)

        # Build prompt - EXACT format from rag_setup.py
        prompt = f"""Context:
{context_text}

Question: {question}

Answer:"""

        # Generate answer
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if use_quantization:
            inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer (remove prompt)
        if prompt in answer:
            answer = answer.replace(prompt, "").strip()
        else:
            # Fallback: find where answer starts
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

        return answer, context_docs, context_scores

    # Run tests
    logger.info(f"\n=== Running Tests ===")
    logger.info(f"Testing {len(test_questions)} questions...")

    results = {
        "experiment_type": "model_test_local",
        "timestamp": datetime.now().isoformat(),
        "reproducibility": {
            "random_seed": get_tuning_config().random_seed,
            "system_metadata": get_system_metadata()
        },
        "model": {
            "path": str(model_path_abs),
            "stage": stage,
            "version": model_version,
            "checkpoint": checkpoint_name,
            "provider": "transformers_4bit",
            "base_model": base_model_name,
            "lora_r": adapter_config.get('r'),
            "lora_alpha": adapter_config.get('lora_alpha')
        },
        "config": {
            "top_k": TOP_K,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "quantization": "4bit_nf4" if use_quantization else "none"
        },
        "dataset": {
            "documents_folder": documents_folder,
            "num_documents": len(documents),
            "test_questions": test_questions
        },
        "questions": test_questions,
        "answers": [],
        "context_scores": [],
        "response_times": [],
        "context_docs": [],
        "answer_quality_scores": []
    }

    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n[{i}/{len(test_questions)}] {question[:60]}...")

        start_time = time.time()
        try:
            answer, context_docs, context_scores = query_rag(question)
            elapsed = time.time() - start_time

            # Evaluate answer quality
            context_text = "\n\n".join(context_docs)
            quality_scores = evaluate_answer_quality(question, answer, context_text)

            results["answers"].append(answer)
            results["context_scores"].append(context_scores)
            results["response_times"].append(elapsed)
            results["context_docs"].append(context_docs)
            results["answer_quality_scores"].append(quality_scores)

            avg_context = sum(context_scores) / len(context_scores) if context_scores else 0.0
            logger.info(f"  ✓ Length: {len(answer)} chars")
            logger.info(f"    Context: {avg_context:.3f}, Quality: {quality_scores['overall_score']:.3f}")
            logger.info(f"    Time: {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            results["answers"].append(f"ERROR: {str(e)}")
            results["context_scores"].append([])
            results["response_times"].append(0.0)
            results["context_docs"].append([])
            results["answer_quality_scores"].append({
                "length_score": 0.0,
                "context_relevance": 0.0,
                "not_refusal": 0.0,
                "question_relevance": 0.0,
                "overall_score": 0.0
            })

    # Cleanup model to free VRAM
    logger.info("\n=== Cleaning up ===")
    del model
    del tokenizer
    if use_quantization:
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("✓ Model unloaded, VRAM freed")

    # Calculate summary
    all_scores = [score for scores in results["context_scores"] if scores for score in scores]

    quality_metrics = {}
    if results["answer_quality_scores"]:
        metric_keys = ["length_score", "context_relevance", "not_refusal", "question_relevance", "overall_score"]
        for key in metric_keys:
            values = [q[key] for q in results["answer_quality_scores"] if isinstance(q, dict)]
            quality_metrics[f"avg_{key}"] = sum(values) / len(values) if values else 0.0

    results["summary"] = {
        "avg_context_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "avg_response_time": sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0.0,
        "total_questions": len(test_questions),
        "successful_queries": len([a for a in results["answers"] if not a.startswith("ERROR")]),
        **quality_metrics
    }

    logger.info("\n=== Test Complete ===")
    logger.info(f"Avg context score: {results['summary']['avg_context_score']:.3f}")
    logger.info(f"Avg answer quality: {results['summary'].get('avg_overall_score', 0.0):.3f}")
    logger.info(f"Avg response time: {results['summary']['avg_response_time']:.2f}s")
    logger.info(f"Successful: {results['summary']['successful_queries']}/{results['summary']['total_questions']}")

    # Save results
    if output_file:
        if use_timestamp and not has_timestamp(output_file):
            base_name = Path(output_file).stem
            output_file = get_timestamped_filename(base_name, "json")

        output_path = Path(__file__).parent / "results" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Results saved to {output_path}")

        results["_saved_filename"] = str(output_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test model locally with transformers + 4-bit quantization")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to LoRA adapter directory")
    parser.add_argument("--stage", type=str, required=True, choices=["stage_1", "stage_2"],
                       help="Stage name (stage_1 or stage_2)")
    parser.add_argument("--documents", type=str, default="../../data/documents",
                       help="Path to documents folder")
    parser.add_argument("--output", type=str, default="model_test_local_results.json",
                       help="Output filename (default: model_test_local_results.json, auto-timestamped)")
    parser.add_argument("--no-timestamp", action="store_true",
                       help="Don't add timestamp to filename")

    args = parser.parse_args()

    try:
        results = test_model_local(
            model_path=args.model_path,
            stage=args.stage,
            documents_folder=args.documents,
            output_file=args.output,
            use_timestamp=not args.no_timestamp
        )
        print("\n=== Model Test Complete ===")
        saved_file = results.get("_saved_filename", args.output)
        print(f"Results saved to: {saved_file}")
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
