"""
Local RAG Tester
Main orchestration class for testing fine-tuned models locally
Uses ChromaDB for vector storage and transformers + PEFT with 4-bit quantization
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Tuple

# Initialize IMPORT_ERROR at module level to avoid undefined variable errors
IMPORT_ERROR = ""

try:
    import torch
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Constants for magic numbers
PREVIEW_QUESTION_LENGTH = 60
PREVIEW_ANSWER_LENGTH = 80
MAX_TOKENIZER_LENGTH = 2048
VALID_STAGES = {"stage_1", "stage_2"}
ERROR_PREFIX = "ERROR:"

from ...config import get_rag_config, get_tuning_config, RAGConfig
from ...logging_config import setup_logging, get_logger
from ..utils import (
    load_documents_from_folder,
    get_test_questions,
    evaluate_answer_quality,
    get_timestamped_filename,
    has_timestamp
)
from .chromadb_store import ChromaDBStore
from .model_loader import ModelLoader
from .rag_query import RAGQuery
from .comparison import compare_results

# Setup logging
setup_logging()
logger = get_logger(__name__)


class LocalRAGTester:
    """
    Local RAG testing class for evaluating fine-tuned models
    
    Uses ChromaDB for vector storage and transformers + PEFT
    with 4-bit quantization for efficient model loading and inference.
    """
    
    def __init__(
        self,
        documents_folder: str,
        test_questions: Optional[List[str]] = None,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize LocalRAGTester
        
        Args:
            documents_folder: Path to folder containing documents
            test_questions: List of test questions (uses default if None)
            config: RAGConfig instance (uses default if None)
            
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If documents_folder is invalid or test_questions is empty
        """
        if not DEPS_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {IMPORT_ERROR}")
        
        # Input validation
        if not documents_folder or not isinstance(documents_folder, str):
            raise ValueError(f"documents_folder must be a non-empty string, got: {type(documents_folder)}")
        
        documents_folder_path = Path(documents_folder)
        if not documents_folder_path.exists():
            raise ValueError(f"documents_folder does not exist: {documents_folder}")
        if not documents_folder_path.is_dir():
            raise ValueError(f"documents_folder must be a directory: {documents_folder}")
        
        self.config = config or get_rag_config()
        self.documents_folder = documents_folder
        self.test_questions = test_questions or get_test_questions()
        
        # Validate test_questions
        if not self.test_questions:
            raise ValueError("test_questions cannot be empty. Provide questions or ensure defaults are available.")
        if not isinstance(self.test_questions, list):
            raise ValueError(f"test_questions must be a list, got: {type(self.test_questions)}")
        if not all(isinstance(q, str) and q.strip() for q in self.test_questions):
            raise ValueError("test_questions must be a list of non-empty strings")
        
        self.vector_store = ChromaDBStore(
            collection_name="model_test",
            embedding_model=self.config.embedding_model
        )
        
        self._load_documents()
        
        logger.info(f"✓ LocalRAGTester initialized")
        logger.info(f"  Documents: {len(self.documents)}")
        logger.info(f"  Questions: {len(self.test_questions)}")
        logger.info(f"  Embedding model: {self.config.embedding_model}")
        logger.info(f"  Retrieval: top_k={self.config.top_k}")
        logger.info(f"  Generation: temperature={self.config.temperature}, max_tokens={self.config.local_testing_max_tokens}")
    
    def _load_documents(self):
        """Load documents from folder and index in ChromaDB"""
        logger.info(f"Loading documents from {self.documents_folder}...")
        
        self.documents = load_documents_from_folder(self.documents_folder, include_html=True)
        
        if not self.documents:
            raise ValueError(f"No documents found in {self.documents_folder}")
        
        logger.info(f"✓ Loaded {len(self.documents)} documents")
        
        self.vector_store.add_documents(self.documents)
    
    def _run_test_loop(
        self,
        results: Dict[str, Any],
        model: Any,
        tokenizer: Any,
        query_func: Callable[[str, Any, Any, Any, int, float, int], Tuple[str, List[str], List[float]]],
        show_preview: bool = False,
        show_traceback: bool = False
    ) -> None:
        """
        Run test loop for all questions (shared between baseline and fine-tuned)
        
        Args:
            results: Results dictionary to populate
            model: Loaded model
            tokenizer: Loaded tokenizer
            query_func: Function to call for querying (baseline or fine-tuned)
            show_preview: Whether to show answer preview in logs
            show_traceback: Whether to show full traceback on errors
        """
        for i, question in enumerate(self.test_questions, 1):
            question_preview = question[:PREVIEW_QUESTION_LENGTH] + "..." if len(question) > PREVIEW_QUESTION_LENGTH else question
            logger.info(f"[{i}/{len(self.test_questions)}] {question_preview}")
            
            start_time = time.time()
            try:
                answer, context_docs, context_scores = query_func(
                    question, model, tokenizer, self.vector_store,
                    top_k=self.config.top_k,
                    temperature=self.config.temperature,
                    max_tokens=self.config.local_testing_max_tokens
                )
                elapsed = time.time() - start_time
                
                context_text = "\n\n".join(context_docs)
                quality_scores = evaluate_answer_quality(question, answer, context_text)
                
                results["questions"].append(question)
                results["answers"].append(answer)
                results["response_times"].append(elapsed)
                results["context_scores"].append(context_scores)
                results["context_docs"].append(context_docs)
                results["answer_quality_scores"].append(quality_scores)
                
                avg_context = sum(context_scores) / len(context_scores) if context_scores else 0.0
                logger.info(
                    f"  ✓ Answer: {len(answer)} chars, Context: {avg_context:.3f}, "
                    f"Quality: {quality_scores['overall_score']:.3f}, Time: {elapsed:.1f}s"
                )
                if show_preview:
                    answer_preview = answer[:PREVIEW_ANSWER_LENGTH] + "..." if len(answer) > PREVIEW_ANSWER_LENGTH else answer
                    logger.info(f"    Preview: {answer_preview}")
                
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
                if show_traceback:
                    import traceback
                    traceback.print_exc()
                
                results["questions"].append(question)
                results["answers"].append(f"{ERROR_PREFIX} {str(e)}")
                results["response_times"].append(0.0)
                results["context_scores"].append([])
                results["context_docs"].append([])
                results["answer_quality_scores"].append({
                    "length_score": 0.0,
                    "context_relevance": 0.0,
                    "not_refusal": 0.0,
                    "question_relevance": 0.0,
                    "overall_score": 0.0
                })
    
    def _calculate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary metrics from results (shared between baseline and fine-tuned)
        
        Args:
            results: Results dictionary with questions, answers, scores, etc.
            
        Returns:
            Dictionary with summary metrics
        """
        all_scores = [score for scores in results["context_scores"] if scores for score in scores]
        
        quality_metrics = {}
        if results["answer_quality_scores"]:
            metric_keys = ["length_score", "context_relevance", "not_refusal", "question_relevance", "overall_score"]
            for key in metric_keys:
                values = [q[key] for q in results["answer_quality_scores"] if isinstance(q, dict)]
                quality_metrics[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
        
        # Validate results structure before calculating summary
        if not results.get("response_times"):
            logger.warning("No response times found in results, using default values")
        
        # Improved error detection: check for ERROR_PREFIX or empty answers
        successful_queries = sum(
            1 for a in results.get("answers", [])
            if isinstance(a, str) and not a.startswith(ERROR_PREFIX) and a.strip()
        )
        
        return {
            "avg_context_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "avg_response_time": sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0.0,
            "total_questions": len(self.test_questions),
            "successful_queries": successful_queries,
            **quality_metrics
        }
    
    def _find_compatible_baseline(
        self,
        base_model_name: str,
        documents_folder: str,
        num_documents: int,
        test_questions: List[str],
        max_age_days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Find most recent compatible baseline results
        
        Args:
            base_model_name: Base model name to match
            documents_folder: Documents folder path to match
            num_documents: Number of documents to match
            test_questions: List of test questions to match
            max_age_days: Maximum age in days for reusable baseline
            
        Returns:
            Baseline results dictionary if found and compatible, None otherwise
        """
        baseline_results_dir = Path(__file__).parent.parent / "baseline" / "results"
        
        if not baseline_results_dir.exists():
            return None
        
        # Find all baseline result files, sorted by modification time (newest first)
        baseline_files = sorted(
            baseline_results_dir.glob("baseline_results_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not baseline_files:
            return None
        
        # Check age threshold
        from datetime import timedelta
        age_threshold = datetime.now() - timedelta(days=max_age_days)
        
        for baseline_file in baseline_files:
            try:
                # Check file age
                file_time = datetime.fromtimestamp(baseline_file.stat().st_mtime)
                if file_time < age_threshold:
                    logger.info(f"Baseline {baseline_file.name} is too old ({file_time.date()}), skipping")
                    continue
                
                # Load and validate baseline
                try:
                    with open(baseline_file, 'r', encoding='utf-8') as f:
                        baseline = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Error parsing JSON in baseline {baseline_file.name}: {e}, skipping")
                    continue
                except OSError as e:
                    logger.warning(f"Error reading baseline file {baseline_file.name}: {e}, skipping")
                    continue
                
                # Validate baseline structure
                if not isinstance(baseline, dict):
                    logger.debug(f"Baseline {baseline_file.name} is not a valid dictionary, skipping")
                    continue
                
                # Validate compatibility
                baseline_config = baseline.get("config", {})
                baseline_dataset = baseline.get("dataset", {})
                baseline_questions = baseline.get("questions", [])
                
                # Validate baseline has required structure
                if not isinstance(baseline_config, dict) or not isinstance(baseline_dataset, dict):
                    logger.debug(f"Baseline {baseline_file.name} has invalid structure, skipping")
                    continue
                
                if not isinstance(baseline_questions, list):
                    logger.debug(f"Baseline {baseline_file.name} has invalid questions format, skipping")
                    continue
                
                # Check model name
                if baseline_config.get("model_name") != base_model_name:
                    logger.debug(f"Baseline {baseline_file.name} has different model, skipping")
                    continue
                
                # Check documents (normalize paths for comparison)
                try:
                    baseline_docs_folder = str(Path(baseline_dataset.get("documents_folder", "")).resolve())
                    current_docs_folder = str(Path(documents_folder).resolve())
                except (OSError, ValueError) as e:
                    logger.debug(f"Error resolving paths for baseline {baseline_file.name}: {e}, skipping")
                    continue
                if baseline_docs_folder != current_docs_folder:
                    logger.debug(f"Baseline {baseline_file.name} has different documents folder, skipping")
                    continue
                
                # Check number of documents
                if baseline_dataset.get("num_documents") != num_documents:
                    logger.debug(f"Baseline {baseline_file.name} has different number of documents, skipping")
                    continue
                
                # Check test questions (must match exactly)
                if baseline_questions != test_questions:
                    logger.debug(f"Baseline {baseline_file.name} has different test questions, skipping")
                    continue
                
                # Check config parameters (top_k, temperature, max_tokens, embedding_model)
                # Note: Old baseline results may not have max_tokens/embedding_model,
                # so they'll be skipped automatically. Only new HuggingFace baselines will match.
                if (baseline_config.get("top_k") != self.config.top_k or
                    baseline_config.get("temperature") != self.config.temperature):
                    logger.debug(f"Baseline {baseline_file.name} has different config (top_k/temperature), skipping")
                    continue
                
                # Check for new baseline format (must have max_tokens and embedding_model)
                if (baseline_config.get("max_tokens") != self.config.local_testing_max_tokens or
                    baseline_config.get("embedding_model") != self.config.embedding_model):
                    logger.debug(f"Baseline {baseline_file.name} has different config (max_tokens/embedding_model) or is old format, skipping")
                    continue
                
                # Found compatible baseline!
                logger.info(f"✓ Found compatible baseline: {baseline_file.name} (from {file_time.strftime('%Y-%m-%d %H:%M:%S')})")
                return baseline
                
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Error reading baseline {baseline_file.name}: {e}, skipping")
                continue
        
        return None
    
    def _save_results(
        self,
        results: Dict[str, Any],
        output_file: str,
        output_subdir: str,
        use_timestamp: bool = True
    ) -> None:
        """
        Save results to JSON file (shared between baseline and fine-tuned)
        
        Args:
            results: Results dictionary to save
            output_file: Output filename
            output_subdir: Subdirectory name ("baseline" or "simrag")
            use_timestamp: Add timestamp to filename
        """
        if use_timestamp and not has_timestamp(output_file):
            base_name = Path(output_file).stem
            output_file = get_timestamped_filename(base_name, "json")
        
        output_path = Path(__file__).parent.parent / output_subdir / "results" / output_file
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create output directory {output_path.parent}: {e}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"\n✓ Results saved to {output_path}")
            results["_saved_filename"] = str(output_path)
        except (OSError, IOError) as e:
            raise IOError(f"Failed to write results to {output_path}: {e}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize results to JSON: {e}")
    
    def test_baseline_model(
        self,
        base_model_name: str,
        output_file: Optional[str] = None,
        use_timestamp: bool = True
    ) -> Dict[str, Any]:
        """
        Test baseline model
        
        Args:
            base_model_name: HuggingFace model name
            output_file: Optional output filename
            use_timestamp: Add timestamp to filename
            
        Returns:
            Dictionary with baseline test results
            
        Raises:
            ValueError: If base_model_name is invalid
        """
        # Input validation
        if not base_model_name or not isinstance(base_model_name, str) or not base_model_name.strip():
            raise ValueError(f"base_model_name must be a non-empty string, got: {base_model_name}")
        
        logger.info("=" * 60)
        logger.info("BASELINE TESTS (Base Model)")
        logger.info("=" * 60)
        
        model, tokenizer = ModelLoader.load_baseline_model(base_model_name)
        
        results = {
            "experiment_type": "baseline",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model_name": base_model_name,
                "top_k": self.config.top_k,
                "temperature": self.config.temperature,
                "max_tokens": self.config.local_testing_max_tokens,
                "embedding_model": self.config.embedding_model
            },
            "dataset": {
                "documents_folder": self.documents_folder,
                "num_documents": len(self.documents)
            },
            "questions": [],
            "answers": [],
            "response_times": [],
            "context_scores": [],
            "context_docs": [],
            "answer_quality_scores": []
        }
        
        logger.info(f"\nRunning baseline tests on {len(self.test_questions)} questions...\n")
        
        # Run test loop using helper method
        self._run_test_loop(
            results=results,
            model=model,
            tokenizer=tokenizer,
            query_func=RAGQuery.query_baseline,
            show_preview=False,
            show_traceback=False
        )
        
        # Validate results before calculating summary
        if not results.get("response_times"):
            logger.warning("No response times recorded in baseline test results")
        if not results.get("answers"):
            logger.warning("No answers recorded in baseline test results")
        
        # Calculate summary using helper method
        results["summary"] = self._calculate_summary(results)
        
        logger.info("\n=== Baseline Tests Complete ===")
        logger.info(f"Avg context score: {results['summary']['avg_context_score']:.3f}")
        logger.info(f"Avg answer quality: {results['summary'].get('avg_overall_score', 0.0):.3f}")
        logger.info(f"Avg response time: {results['summary']['avg_response_time']:.2f}s")
        
        # Clean up baseline model to free VRAM
        ModelLoader.cleanup_model(model, tokenizer)
        
        if output_file:
            self._save_results(results, output_file, "baseline", use_timestamp)
        
        return results
    
    def test_finetuned_model(
        self,
        adapter_path: str,
        base_model_name: str,
        stage: str,
        model_version: str = "v1.0",
        checkpoint_name: Optional[str] = None,
        output_file: Optional[str] = None,
        use_timestamp: bool = True
    ) -> Dict[str, Any]:
        """
        Test fine-tuned model
        
        Args:
            adapter_path: Path to LoRA adapter directory
            base_model_name: HuggingFace model name
            stage: Stage name ("stage_1" or "stage_2")
            model_version: Model version (e.g., "v1.8")
            checkpoint_name: Optional checkpoint name
            output_file: Optional output filename
            use_timestamp: Add timestamp to filename
            
        Returns:
            Dictionary with fine-tuned test results
        """
        logger.info("\n" + "=" * 60)
        logger.info("FINE-TUNED MODEL TESTS")
        logger.info("=" * 60)
        logger.info("Testing fine-tuned model with identical conditions as baseline...")
        
        # Input validation
        if not adapter_path or not isinstance(adapter_path, str) or not adapter_path.strip():
            raise ValueError(f"adapter_path must be a non-empty string, got: {adapter_path}")
        
        if not base_model_name or not isinstance(base_model_name, str) or not base_model_name.strip():
            raise ValueError(f"base_model_name must be a non-empty string, got: {base_model_name}")
        
        if not stage or not isinstance(stage, str) or stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}, got: {stage}")
        
        adapter_path_obj = Path(adapter_path).resolve()
        
        if not adapter_path_obj.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
        if not adapter_path_obj.is_dir():
            raise ValueError(f"adapter_path must be a directory: {adapter_path}")
        
        adapter_config_file = adapter_path_obj / "adapter_config.json"
        
        if not adapter_config_file.exists():
            raise FileNotFoundError(f"Adapter config not found: {adapter_config_file}")
        
        try:
            with open(adapter_config_file, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in adapter config file {adapter_config_file}: {e}")
        except OSError as e:
            raise IOError(f"Failed to read adapter config file {adapter_config_file}: {e}")
        
        if not isinstance(adapter_config, dict):
            raise ValueError(f"Adapter config must be a dictionary, got: {type(adapter_config)}")
        
        model, tokenizer = ModelLoader.load_finetuned_model(adapter_path, base_model_name)
        
        # Convert absolute path to relative path for anonymization
        from ..utils import get_relative_path
        relative_path = get_relative_path(str(adapter_path_obj))
        
        results = {
            "experiment_type": "model_test",
            "timestamp": datetime.now().isoformat(),
            "model": {
                "path": relative_path,
                "stage": stage,
                "version": model_version,
                "checkpoint": checkpoint_name,
                "provider": "transformers_4bit",
                "base_model": base_model_name,
                "model_size": "small" if "1.5B" in base_model_name or "1B" in base_model_name else "medium",
                "lora_r": adapter_config.get('r'),
                "lora_alpha": adapter_config.get('lora_alpha')
            },
            "config": {
                "top_k": self.config.top_k,
                "temperature": self.config.temperature,
                "max_tokens": self.config.local_testing_max_tokens,
                "embedding_model": self.config.embedding_model
            },
            "dataset": {
                "documents_folder": self.documents_folder,
                "num_documents": len(self.documents)
            },
            "questions": [],
            "answers": [],
            "response_times": [],
            "context_scores": [],
            "context_docs": [],
            "answer_quality_scores": []
        }
        
        logger.info(f"\nRunning tests on {len(self.test_questions)} questions...\n")
        
        # Run test loop using helper method
        self._run_test_loop(
            results=results,
            model=model,
            tokenizer=tokenizer,
            query_func=RAGQuery.query_finetuned,
            show_preview=True,
            show_traceback=True
        )
        
        # Validate results before calculating summary
        if not results.get("response_times"):
            logger.warning("No response times recorded in fine-tuned test results")
        if not results.get("answers"):
            logger.warning("No answers recorded in fine-tuned test results")
        
        # Calculate summary using helper method
        results["summary"] = self._calculate_summary(results)
        
        logger.info("\n=== Fine-Tuned Model Tests Complete ===")
        logger.info(f"Avg context score: {results['summary']['avg_context_score']:.3f}")
        logger.info(f"Avg answer quality: {results['summary'].get('avg_overall_score', 0.0):.3f}")
        logger.info(f"Avg response time: {results['summary']['avg_response_time']:.2f}s")
        logger.info(f"Successful: {results['summary']['successful_queries']}/{results['summary']['total_questions']}")
        
        # Clean up fine-tuned model
        ModelLoader.cleanup_model(model, tokenizer)
        
        if output_file:
            self._save_results(results, output_file, "simrag", use_timestamp)
        
        return results
    
    def compare_results(
        self,
        baseline_results: Dict[str, Any],
        finetuned_results: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare baseline vs fine-tuned results
        
        Args:
            baseline_results: Baseline test results
            finetuned_results: Fine-tuned test results
            output_file: Optional output filename
            
        Returns:
            Dictionary with comparison results
            
        Raises:
            ValueError: If results structures are invalid or incompatible
        """
        # Validate input structures
        if not isinstance(baseline_results, dict):
            raise ValueError(f"baseline_results must be a dictionary, got: {type(baseline_results)}")
        if not isinstance(finetuned_results, dict):
            raise ValueError(f"finetuned_results must be a dictionary, got: {type(finetuned_results)}")
        
        # Validate required keys exist
        required_keys = ["questions", "summary", "context_scores"]
        for key in required_keys:
            if key not in baseline_results:
                raise ValueError(f"baseline_results missing required key: {key}")
            if key not in finetuned_results:
                raise ValueError(f"finetuned_results missing required key: {key}")
        
        # Validate question counts match
        baseline_q_count = len(baseline_results.get("questions", []))
        finetuned_q_count = len(finetuned_results.get("questions", []))
        if baseline_q_count != finetuned_q_count:
            logger.warning(
                f"Question count mismatch: baseline has {baseline_q_count}, "
                f"fine-tuned has {finetuned_q_count}. Comparison may be inaccurate."
            )
        
        # Validate summary structure
        if "summary" not in baseline_results or not isinstance(baseline_results["summary"], dict):
            raise ValueError("baseline_results must have a valid 'summary' dictionary")
        if "summary" not in finetuned_results or not isinstance(finetuned_results["summary"], dict):
            raise ValueError("finetuned_results must have a valid 'summary' dictionary")
        
        return compare_results(baseline_results, finetuned_results, output_file)
    
    def run_full_test(
        self,
        base_model_name: str,
        adapter_path: str,
        stage: str,
        model_version: str = "v1.0",
        checkpoint_name: Optional[str] = None,
        baseline_output: Optional[str] = None,
        finetuned_output: Optional[str] = None,
        comparison_output: Optional[str] = None,
        use_timestamp: bool = True
    ) -> Dict[str, Any]:
        """
        Run full test flow: baseline → fine-tuned → compare
        
        Args:
            base_model_name: HuggingFace model name
            adapter_path: Path to LoRA adapter directory
            stage: Stage name ("stage_1" or "stage_2")
            model_version: Model version (e.g., "v1.8")
            checkpoint_name: Optional checkpoint name
            baseline_output: Optional baseline results filename
            finetuned_output: Optional fine-tuned results filename
            comparison_output: Optional comparison results filename
            use_timestamp: Add timestamp to filenames
            
        Returns:
            Dictionary with all results (baseline, fine-tuned, comparison)
        """
        logger.info("=" * 60)
        logger.info("FULL TEST FLOW (Baseline → Fine-Tuned → Compare)")
        logger.info("=" * 60)
        
        # Step 1: Test baseline (or reuse if compatible)
        baseline_reused = False
        if self.config.reuse_baseline:
            logger.info("Checking for compatible baseline results...")
            compatible_baseline = self._find_compatible_baseline(
                base_model_name=base_model_name,
                documents_folder=self.documents_folder,
                num_documents=len(self.documents),
                test_questions=self.test_questions,
                max_age_days=self.config.baseline_max_age_days
            )
            
            if compatible_baseline:
                baseline_results = compatible_baseline.copy()
                baseline_results["_reused"] = True  # Mark as reused for CLI display
                baseline_reused = True
                logger.info("✓ Reusing existing baseline (saves ~5-10 minutes)")
                logger.info(f"  Baseline from: {compatible_baseline.get('timestamp', 'unknown')}")
            else:
                logger.info("No compatible baseline found, running new baseline tests...")
                baseline_results = self.test_baseline_model(
                    base_model_name=base_model_name,
                    output_file=baseline_output or "baseline_results.json",
                    use_timestamp=use_timestamp
                )
        else:
            logger.info("Baseline reuse disabled, running new baseline tests...")
            baseline_results = self.test_baseline_model(
                base_model_name=base_model_name,
                output_file=baseline_output or "baseline_results.json",
                use_timestamp=use_timestamp
            )
        
        # Step 2: Test fine-tuned model
        finetuned_results = self.test_finetuned_model(
            adapter_path=adapter_path,
            base_model_name=base_model_name,
            stage=stage,
            model_version=model_version,
            checkpoint_name=checkpoint_name,
            output_file=finetuned_output or "simrag_results.json",
            use_timestamp=use_timestamp
        )
        
        # Step 3: Compare results
        comparison_results = self.compare_results(
            baseline_results=baseline_results,
            finetuned_results=finetuned_results,
            output_file=comparison_output or "comparison_results.json"
        )
        
        return {
            "baseline": baseline_results,
            "finetuned": finetuned_results,
            "comparison": comparison_results
        }

