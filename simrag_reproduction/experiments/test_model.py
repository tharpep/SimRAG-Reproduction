"""
Test Any Model
Test a specific Stage 1 or Stage 2 model with the RAG system
"""

import json
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..rag.rag_setup import BasicRAG
from ..config import get_rag_config, get_tuning_config
from ..tuning.model_registry import get_model_registry
from ..logging_config import setup_logging, get_logger
from .utils import load_documents_from_folder, get_test_questions

try:
    import torch
except ImportError:
    torch = None  # type: ignore

# Setup logging
setup_logging()
logger = get_logger(__name__)


def list_available_models(stage: str, model_size: str = "small") -> List[Dict[str, Any]]:
    """
    List all available models for a given stage, including checkpoints
    
    Args:
        stage: Stage name ("stage_1" or "stage_2")
        model_size: Model size ("small" or "medium")
        
    Returns:
        List of model info dictionaries with version, path, checkpoint info, and metadata
    """
    model_suffix = "1b" if model_size == "small" else "8b"
    stage_dir = Path(f"./tuned_models/model_{model_suffix}/{stage}")
    
    if not stage_dir.exists():
        return []
    
    models = []
    config = get_tuning_config()
    config.model_size = model_size
    registry = get_model_registry(config)
    
    # Get all versions from registry
    all_versions = registry.get_all_versions()
    version_metadata = {v.version: v for v in all_versions}
    
    # Scan all version directories
    for model_dir in stage_dir.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("v"):
            version = model_dir.name
            version_info = version_metadata.get(version)
            
            # Check for adapter files in version directory and checkpoints
            adapter_files = list(model_dir.glob("**/adapter_model.safetensors")) + \
                           list(model_dir.glob("**/adapter_model.bin"))
            
            if adapter_files:
                # Group by checkpoint directory
                checkpoint_paths = {}
                version_path = None
                
                for adapter_file in adapter_files:
                    parent = adapter_file.parent
                    if "checkpoint" in parent.name:
                        checkpoint_paths[parent.name] = str(parent)
                    elif parent == model_dir:
                        version_path = str(parent)
                
                # Add version-level model if exists
                if version_path:
                    models.append({
                        "version": version,
                        "checkpoint": None,
                        "path": version_path,
                        "display_name": f"{version} (final)",
                        "created_at": version_info.created_at if version_info else "Unknown",
                        "training_time": version_info.training_time_seconds if version_info else None,
                        "final_loss": version_info.final_loss if version_info else None,
                        "notes": version_info.notes if version_info else None,
                        "experiment_run_id": version_info.experiment_run_id if version_info else None
                    })
                
                # Add checkpoint models
                for checkpoint_name, checkpoint_path in sorted(checkpoint_paths.items()):
                    # Extract step number from checkpoint name (e.g., "checkpoint-500" -> 500)
                    step_num = None
                    try:
                        step_num = int(checkpoint_name.split("-")[-1])
                    except (ValueError, IndexError):
                        pass
                    
                    models.append({
                        "version": version,
                        "checkpoint": checkpoint_name,
                        "path": checkpoint_path,
                        "display_name": f"{version} ({checkpoint_name})",
                        "step": step_num,
                        "created_at": version_info.created_at if version_info else "Unknown",
                        "training_time": version_info.training_time_seconds if version_info else None,
                        "final_loss": version_info.final_loss if version_info else None,
                        "notes": version_info.notes if version_info else None,
                        "experiment_run_id": version_info.experiment_run_id if version_info else None
                    })
    
    # Sort by version, then by checkpoint step (checkpoints first, then final)
    def sort_key(m):
        try:
            version_num = float(m["version"][1:])
        except (ValueError, IndexError):
            version_num = 0.0
        
        # Sort: version (desc), then checkpoint step (desc, None last)
        checkpoint_step = m.get("step") if m.get("checkpoint") else float('inf')
        return (-version_num, checkpoint_step)
    
    models.sort(key=sort_key)
    return models


def test_model(
    model_path: str,
    stage: str,
    documents_folder: str = "../../data/documents",
    test_questions: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    use_timestamp: bool = True
) -> Dict[str, Any]:
    """
    Test a specific model with the RAG system
    
    Args:
        model_path: Path to the model directory
        stage: Stage name ("stage_1" or "stage_2")
        documents_folder: Path to documents folder
        test_questions: List of test questions (uses default if None)
        output_file: Path to save results JSON (optional)
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"=== Testing Model: {model_path} ===")
    
    if test_questions is None:
        test_questions = get_test_questions()
    
    # Load documents
    logger.info(f"Loading documents from {documents_folder}...")
    documents = load_documents_from_folder(documents_folder, include_html=True)
    
    if not documents:
        raise ValueError(f"No documents found in {documents_folder}")
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Get Ollama model name for this model
    rag_config = get_rag_config()
    model_path_obj = Path(model_path)
    
    # Extract version and checkpoint from path
    # Path structure: tuned_models/model_1b/stage_1/v1.8/checkpoint-500/
    model_version = model_path_obj.name
    checkpoint_name = None
    
    if "checkpoint" in model_version:
        # If we're in a checkpoint directory, extract checkpoint name and version
        checkpoint_name = model_version
        model_version = model_path_obj.parent.name
    elif model_path_obj.parent.name in ["stage_1", "stage_2"]:
        # We're already at the version level (final model, no checkpoint)
        model_version = model_path_obj.name
    else:
        # Check if parent is a checkpoint directory
        parent_name = model_path_obj.parent.name
        if "checkpoint" in parent_name:
            checkpoint_name = parent_name
            model_version = model_path_obj.parent.parent.name
    
    # Resolve model path to absolute path
    model_path_abs = Path(model_path).resolve()
    
    # Verify adapter files exist
    adapter_file = model_path_abs / "adapter_model.safetensors"
    adapter_config = model_path_abs / "adapter_config.json"
    
    if not adapter_file.exists():
        raise Exception(f"Adapter file not found: {adapter_file}")
    if not adapter_config.exists():
        raise Exception(f"Adapter config not found: {adapter_config}")
    
    # Convert LoRA adapters to GGUF for Ollama
    # GGUF models load 100% on GPU, unlike merged FP16 models which get split CPU/GPU
    gguf_path = model_path_abs / f"{model_path_abs.name}_q4_k_m.gguf"

    if not gguf_path.exists():
        logger.info("Converting LoRA adapters to GGUF format (one-time operation)...")
        logger.info("This requires merging adapters and converting to GGUF.")
        logger.info("This will take 5-10 minutes but only happens once per checkpoint.")

        # First, merge adapters if not already merged
        merged_model_dir = model_path_abs / "merged"
        if not merged_model_dir.exists():
            logger.info("Step 1/2: Merging LoRA adapters...")
            from ..tuning.adapter_merge import merge_lora_adapters

            try:
                merge_lora_adapters(
                    adapter_path=str(model_path_abs),
                    output_path=str(merged_model_dir)
                )
                logger.info(f"✓ Merged model saved to {merged_model_dir}")
            except Exception as e:
                raise Exception(f"Failed to merge adapters: {e}")
        else:
            logger.info(f"Using existing merged model from {merged_model_dir}")

        # Then convert to GGUF
        logger.info("Step 2/2: Converting to GGUF (Q4_K_M quantization)...")
        from ..tuning.gguf_convert import convert_to_gguf

        try:
            gguf_path_str = convert_to_gguf(
                model_path=str(merged_model_dir),
                output_path=str(gguf_path),
                quantization="Q4_K_M"
            )
            gguf_path = Path(gguf_path_str)
            logger.info(f"✓ GGUF model saved to {gguf_path}")
        except Exception as e:
            raise Exception(f"Failed to convert to GGUF: {e}")
    else:
        logger.info(f"Using existing GGUF model: {gguf_path}")

    model_to_use = gguf_path
    
    # Generate unique Ollama model name
    def get_ollama_model_name_with_checkpoint(stage: str, version: str, checkpoint: Optional[str], model_size: str) -> str:
        """Generate Ollama model name including checkpoint info"""
        size_suffix = "1b" if model_size == "small" else "7b"
        stage_num = stage.replace("stage_", "")
        version_clean = version.replace(".", "-")
        
        if checkpoint:
            checkpoint_clean = checkpoint.replace("checkpoint-", "ckpt").replace("-", "")
            return f"simrag-{size_suffix}-stage{stage_num}-{version_clean}-{checkpoint_clean}"
        else:
            return f"simrag-{size_suffix}-stage{stage_num}-{version_clean}"
    
    ollama_model_name = get_ollama_model_name_with_checkpoint(
        stage=stage,
        version=model_version,
        checkpoint=checkpoint_name,
        model_size=rag_config.model_size
    )
    
    # Register merged model with Ollama
    model_exists = False
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=5
        )
        if result.returncode == 0 and ollama_model_name in result.stdout:
            model_exists = True
    except Exception:
        pass
    
    if not model_exists:
        logger.info(f"Registering merged model with Ollama: {ollama_model_name}")
    else:
        # Check if we need to recreate with new parameters
        logger.info("Checking if model needs to be recreated with optimizations...")

        # Check current model configuration via `ollama show`
        needs_recreation = False
        try:
            result = subprocess.run(
                ["ollama", "show", ollama_model_name, "--modelfile"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            if result.returncode == 0:
                modelfile_content = result.stdout
                # Check if num_gpu is set correctly (should be 99 or high number for full GPU use)
                if "num_gpu 99" not in modelfile_content and "num_gpu -1" not in modelfile_content:
                    logger.info("Model exists but doesn't have GPU optimizations")
                    needs_recreation = True
        except Exception as e:
            logger.warning(f"Could not check model configuration: {e}")
            needs_recreation = True

        if needs_recreation:
            logger.info("Recreating model with performance optimizations...")
            # Delete existing model
            try:
                subprocess.run(
                    ["ollama", "rm", ollama_model_name],
                    capture_output=True,
                    timeout=10
                )
                model_exists = False
                logger.info(f"Removed old model: {ollama_model_name}")
            except Exception as e:
                logger.warning(f"Could not remove old model: {e}")
    
    if not model_exists:
        logger.info(f"Creating Ollama model: {ollama_model_name}")

        # Determine base model based on model size
        base_model = "qwen2.5:1.5b" if rag_config.model_size == "small" else "qwen2.5:7b"

        # Ensure base model is available in Ollama
        logger.info(f"Ensuring base model {base_model} is available...")
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )
            if base_model not in result.stdout:
                logger.info(f"Pulling base model {base_model}...")
                subprocess.run(
                    ["ollama", "pull", base_model],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=300
                )
        except Exception as e:
            logger.warning(f"Could not check/pull base model: {e}")

        # Create Modelfile for GGUF model
        modelfile_path = model_path_abs / "Modelfile"
        # Use GGUF file (optimized for GPU inference)
        model_abs = Path(model_to_use).resolve()
        model_str = str(model_abs).replace('\\', '/')

        modelfile_content = f"""# Fine-tuned model: {ollama_model_name}
FROM {model_str}

# Performance optimization - force full GPU usage
PARAMETER num_ctx 2048
PARAMETER num_gpu -1
PARAMETER num_thread 8

# Generation parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"

# System message
SYSTEM \"\"\"You are a helpful AI assistant trained to answer questions accurately and concisely based on provided context.\"\"\"
"""
        
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        logger.info(f"Creating Ollama model from GGUF...")
        # GGUF models load 100% on GPU for optimal performance
        import os
        original_dir = os.getcwd()

        try:
            os.chdir(str(model_path_abs))
            result = subprocess.run(
                ["ollama", "create", ollama_model_name, "-f", "Modelfile"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300
            )
        finally:
            os.chdir(original_dir)
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise Exception(f"Failed to create Ollama model: {error_msg}")
        
        logger.info(f"✓ Ollama model created: {ollama_model_name}")
    else:
        logger.info(f"Using existing Ollama model: {ollama_model_name}")
    
    # Initialize RAG with Ollama
    # Increase timeout for first query (model loading)
    import os
    os.environ["OLLAMA_CHAT_TIMEOUT"] = "300.0"  # 5 minutes for first query
    
    rag = BasicRAG(
        collection_name="model_test",
        use_persistent=False,
        force_provider="ollama",
        ollama_model_name=ollama_model_name
    )
    
    rag.add_documents(documents)
    
    # Check GPU VRAM availability
    try:
        if torch and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({total_vram:.1f} GB VRAM)")
    except Exception as e:
        logger.warning(f"Could not check GPU info: {e}")
    
    # Warm up the model with a simple query (first query loads model into memory)
    logger.info("Warming up Ollama model (loading into memory)...")
    logger.info("This may take 1-3 minutes on first run (GPU at 100% is normal)...")
    try:
        start = time.time()
        rag.query("Hello")  # Very short query to minimize warmup time
        elapsed = time.time() - start
        logger.info(f"✓ Model warmed up successfully in {elapsed:.1f}s")
        
        # Verify GPU usage
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if "CPU/GPU" in result.stdout:
            # Extract the CPU/GPU split info
            for line in result.stdout.split('\n'):
                if ollama_model_name in line and "CPU/GPU" in line:
                    logger.info(f"Model loaded: {line.strip()}")
                    # Warn if mostly on CPU
                    if "CPU/GPU" in line:
                        parts = line.split()
                        for part in parts:
                            if "%" in part and "/" in part:
                                cpu_pct = int(part.split('/')[0].replace('%', ''))
                                if cpu_pct > 50:
                                    logger.warning(f"⚠ Model is {cpu_pct}% on CPU - this will be slow!")
                                    logger.warning(f"⚠ Expected: mostly on GPU for your 10GB card")
                                else:
                                    logger.info(f"✓ Model is running primarily on GPU")
                                break
    except Exception as e:
        logger.warning(f"Warmup query failed (will retry on first test question): {e}")
    
    # Reduce timeout for subsequent queries
    os.environ["OLLAMA_CHAT_TIMEOUT"] = "120.0"  # 2 minutes for regular queries
    
    # Test performance - directly run the test loop (don't load another model!)
    logger.info(f"Testing on {len(test_questions)} questions...")
    
    # Import utilities
    from ..experiments.utils import evaluate_answer_quality
    
    results_data = {
        "questions": test_questions,
        "answers": [],
        "response_times": [],
        "context_scores": [],
        "answer_quality_scores": []
    }
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"Testing question {i}/{len(test_questions)}: {question[:50]}...")
        
        try:
            start_time = time.time()
            answer, context_docs, context_scores = rag.query(question)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Evaluate context quality (use first context doc if available)
            context_text = context_docs[0] if context_docs else ""
            context_quality = evaluate_answer_quality(question, answer, context_text)
            
            # Evaluate answer quality
            answer_quality = evaluate_answer_quality(question, answer, "")
            
            # Extract overall scores
            context_score = context_quality.get("overall_score", 0.0)
            quality_score = answer_quality.get("overall_score", 0.0)
            
            results_data["answers"].append(answer)
            results_data["response_times"].append(response_time)
            results_data["context_scores"].append(context_score)
            results_data["answer_quality_scores"].append(quality_score)
            
            logger.info(f"  Answer length: {len(answer)} chars, Context: {context_score:.3f}, Quality: {quality_score:.3f}, Time: {response_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error testing question '{question[:50]}...': {e}")
            results_data["answers"].append(f"ERROR: {str(e)}")
            results_data["response_times"].append(0.0)
            results_data["context_scores"].append(0.0)
            results_data["answer_quality_scores"].append(0.0)
    
    # Calculate summary metrics
    test_results = {
        **results_data,
        "avg_response_time": sum(results_data["response_times"]) / len(test_questions) if test_questions else 0,
        "avg_context_score": sum(results_data["context_scores"]) / len(test_questions) if test_questions else 0,
        "avg_answer_quality": sum(results_data["answer_quality_scores"]) / len(test_questions) if test_questions else 0,
        "total_questions": len(test_questions)
    }
    
    # Format results consistently with other experiments
    results = {
        "experiment_type": "model_test",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "path": model_path,
            "stage": stage,
            "version": model_version,
            "checkpoint": checkpoint_name,
            "ollama_model_name": ollama_model_name,
            "provider": "ollama"
        },
        "dataset": {
            "documents_folder": documents_folder,
            "num_documents": len(documents)
        },
        "questions": test_results["questions"],
        "answers": test_results["answers"],
        "response_times": test_results["response_times"],
        "context_scores": test_results["context_scores"],
        "answer_quality_scores": test_results["answer_quality_scores"],
        "summary": {
            "avg_context_score": test_results["avg_context_score"],
            "avg_response_time": test_results["avg_response_time"],
            "avg_answer_quality": test_results["avg_answer_quality"]
        }
    }
    
    logger.info(f"\nTest completed!")
    logger.info(f"  Avg context score: {results['summary']['avg_context_score']:.3f}")
    logger.info(f"  Avg answer quality: {results['summary']['avg_answer_quality']:.3f}")
    logger.info(f"  Avg response time: {results['summary']['avg_response_time']:.2f}s")
    
    # Save results
    if output_file:
        if use_timestamp:
            from .utils import get_timestamped_filename, has_timestamp
            if not has_timestamp(output_file):
                base_name = Path(output_file).stem
                output_file = get_timestamped_filename(base_name, "json")
        
        output_path = Path(__file__).parent / "results" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
        
        results["_saved_filename"] = str(output_path)
    
    return results


def export_model(
    model_path: str,
    stage: str,
    output_name: Optional[str] = None
) -> str:
    """
    Export a model to a cross-platform ZIP file for Colab
    
    Args:
        model_path: Path to the model directory (checkpoint directory)
        stage: Stage name ("stage_1" or "stage_2")
        output_name: Optional output ZIP filename (auto-generated if None)
        
    Returns:
        Path to the created ZIP file
    """
    logger.info(f"=== Exporting Model: {model_path} ===")
    
    model_path_abs = Path(model_path).resolve()
    
    if not model_path_abs.exists():
        raise ValueError(f"Model path does not exist: {model_path_abs}")
    
    # Verify it's a valid checkpoint/adapter directory
    adapter_files = list(model_path_abs.glob("adapter_model.*"))
    if not adapter_files:
        raise ValueError(f"No adapter files found in {model_path_abs}. Expected adapter_model.safetensors or adapter_model.bin")
    
    # Generate output name if not provided
    if output_name is None:
        # Use checkpoint name with "-fixed" suffix (matching Colab workflow)
        output_name = f"{model_path_abs.name}-fixed.zip"
    
    # Create ZIP in the model's parent directory (same directory as checkpoint)
    output_path = model_path_abs.parent / output_name
    
    # Import and use the zip export utility
    from ..tuning.zip_export import create_cross_platform_zip
    
    zip_path = create_cross_platform_zip(
        source_dir=str(model_path_abs),
        output_path=str(output_path)
    )
    
    logger.info(f"✓ Model exported successfully: {zip_path}")
    return zip_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test a specific model")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--stage", type=str, required=True, choices=["stage_1", "stage_2"],
                       help="Stage name (stage_1 or stage_2)")
    parser.add_argument("--documents", type=str, default="../../data/documents",
                       help="Path to documents folder")
    parser.add_argument("--output", type=str, default="model_test_results.json",
                       help="Output filename (default: model_test_results.json, auto-timestamped)")
    parser.add_argument("--no-timestamp", action="store_true",
                       help="Don't add timestamp to filename")
    
    args = parser.parse_args()
    
    try:
        results = test_model(
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

