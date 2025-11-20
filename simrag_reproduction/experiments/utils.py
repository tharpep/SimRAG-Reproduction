"""
Utility functions for experiments
Helper functions for document loading, HTML extraction, etc.
"""

from pathlib import Path
from typing import List, Dict, Any
import re
import random
import numpy as np
from html.parser import HTMLParser

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML files"""
    
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip_tags = {'script', 'style', 'meta', 'link', 'head'}
        self.in_skip_tag = False
    
    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.skip_tags:
            self.in_skip_tag = True
    
    def handle_endtag(self, tag):
        if tag.lower() in self.skip_tags:
            self.in_skip_tag = False
        elif tag.lower() in {'p', 'div', 'br', 'li'}:
            self.text.append('\n')
    
    def handle_data(self, data):
        if not self.in_skip_tag:
            cleaned = data.strip()
            if cleaned:
                self.text.append(cleaned)
    
    def get_text(self) -> str:
        """Get extracted text"""
        text = ' '.join(self.text)
        # Clean up multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


def extract_text_from_html(html_path: Path) -> str:
    """
    Extract text content from HTML file
    
    Args:
        html_path: Path to HTML file
        
    Returns:
        Extracted text content
    """
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        return parser.get_text()
    except Exception as e:
        print(f"Error extracting text from {html_path}: {e}")
        return ""


def load_documents_from_folder(folder_path: str, include_html: bool = True) -> List[str]:
    """
    Load documents from a folder (supports .txt, .md, and optionally .html)
    
    Args:
        folder_path: Path to folder containing documents
        include_html: Whether to include HTML files
        
    Returns:
        List of document texts
    """
    # Resolve to absolute path to handle relative paths correctly
    folder = Path(folder_path).resolve()
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path} (resolved to: {folder})")
    
    documents = []
    
    # Load .txt and .md files
    for ext in ['.txt', '.md']:
        for file_path in folder.glob(f"**/*{ext}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        documents.append(content)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Load HTML files if requested
    if include_html:
        for html_path in folder.glob("**/*.html"):
            text = extract_text_from_html(html_path)
            if text:
                documents.append(text)
    
    return documents


def get_test_questions() -> List[str]:
    """
    Get standard test questions for experiments
    
    Returns:
        List of test questions
    """
    return [
        "What is Docker?",
        "How does CI/CD work?",
        "What is DevOps?",
        "How do you containerize an application?",
        "What is the difference between Docker and virtual machines?",
        "How does Google Cloud Platform work?",
        "What are the benefits of using containers?",
        "Explain the Python standard library.",
        "What is RAG in generative AI?",
        "How do you build a Docker image?"
    ]


def get_timestamped_filename(base_name: str, extension: str = "json") -> str:
    """
    Generate timestamped filename to avoid overwriting previous results
    
    Args:
        base_name: Base filename (e.g., "baseline_results")
        extension: File extension (default: "json")
        
    Returns:
        Timestamped filename (e.g., "baseline_results_2024-11-20_14-30-45.json")
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base_name}_{timestamp}.{extension}"


def has_timestamp(filename: str) -> bool:
    """
    Check if filename already contains a timestamp pattern
    
    Args:
        filename: Filename to check
        
    Returns:
        True if filename appears to have a timestamp
    """
    import re
    # Pattern: YYYY-MM-DD_HH-MM-SS at end of filename
    pattern = r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json$'
    return bool(re.search(pattern, filename))


def evaluate_answer_quality(question: str, answer: str, context: str) -> dict:
    """
    Evaluate answer quality using rule-based metrics
    
    This provides automated scoring for answer quality beyond just context similarity.
    Useful for comparing baseline vs fine-tuned model performance.
    
    Args:
        question: The original question
        answer: The generated answer
        context: The retrieved context documents (concatenated)
        
    Returns:
        Dictionary with quality scores:
        - length_score: 1.0 if reasonable length (20-500 chars), 0.5 if short, 0.0 if too long/empty
        - context_relevance: 0.0-1.0, measures word overlap with context
        - not_refusal: 1.0 if answer attempts to answer, 0.0 if refuses/deflects
        - question_relevance: 0.0-1.0, measures word overlap with question
        - overall_score: Average of all metrics (0.0-1.0)
    """
    scores = {}
    
    # 1. Length scoring (reasonable answer length)
    answer_len = len(answer.strip())
    if answer_len == 0:
        scores['length_score'] = 0.0
    elif answer_len < 20:
        scores['length_score'] = 0.5  # Too short, likely incomplete
    elif answer_len > 500:
        scores['length_score'] = 0.7  # Very long, possibly rambling
    else:
        scores['length_score'] = 1.0  # Good length
    
    # 2. Context relevance (word overlap with context)
    if context:
        # Normalize and tokenize
        context_words = set(w.lower() for w in context.split() if len(w) > 3)
        answer_words = set(w.lower() for w in answer.split() if len(w) > 3)
        
        if len(context_words) > 0 and len(answer_words) > 0:
            overlap = len(context_words & answer_words)
            scores['context_relevance'] = min(1.0, overlap / max(len(answer_words), 5))
        else:
            scores['context_relevance'] = 0.0
    else:
        scores['context_relevance'] = 0.0
    
    # 3. Not a refusal (answer attempts to respond)
    refusal_phrases = [
        "don't know", "do not know", "cannot answer", "can't answer",
        "no information", "not sure", "unclear", "i'm sorry",
        "i apologize", "unable to"
    ]
    answer_lower = answer.lower()
    is_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
    scores['not_refusal'] = 0.0 if is_refusal else 1.0
    
    # 4. Question relevance (answer addresses the question)
    question_words = set(w.lower() for w in question.split() if len(w) > 3)
    answer_words = set(w.lower() for w in answer.split() if len(w) > 3)
    
    if len(question_words) > 0 and len(answer_words) > 0:
        overlap = len(question_words & answer_words)
        scores['question_relevance'] = min(1.0, overlap / max(len(question_words), 3))
    else:
        scores['question_relevance'] = 0.0
    
    # 5. Overall score (average of all metrics)
    metric_values = [
        scores['length_score'],
        scores['context_relevance'],
        scores['not_refusal'],
        scores['question_relevance']
    ]
    scores['overall_score'] = sum(metric_values) / len(metric_values)
    
    return scores


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Sets seeds for Python random, NumPy, and PyTorch (if available).
    This ensures reproducible results across experiment runs.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        # For CUDA reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Additional settings for full reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def validate_experiment_config(
    baseline_questions: List[str],
    simrag_questions: List[str],
    baseline_docs_count: int,
    simrag_docs_count: int
) -> Dict[str, Any]:
    """
    Validate that baseline and SimRAG experiments use the same configuration
    
    Ensures fair comparison by checking:
    - Same test questions
    - Same number of documents
    
    Args:
        baseline_questions: Questions used in baseline experiment
        simrag_questions: Questions used in SimRAG experiment
        baseline_docs_count: Number of documents in baseline
        simrag_docs_count: Number of documents in SimRAG
        
    Returns:
        Dictionary with validation results:
        - is_valid: True if configuration matches
        - issues: List of validation issues found
    """
    issues = []
    
    # Check questions match
    if len(baseline_questions) != len(simrag_questions):
        issues.append(f"Question count mismatch: baseline={len(baseline_questions)}, simrag={len(simrag_questions)}")
    else:
        # Check question content
        baseline_set = set(q.strip().lower() for q in baseline_questions)
        simrag_set = set(q.strip().lower() for q in simrag_questions)
        if baseline_set != simrag_set:
            issues.append("Question content mismatch: different questions used")
    
    # Check document count
    if baseline_docs_count != simrag_docs_count:
        issues.append(f"Document count mismatch: baseline={baseline_docs_count}, simrag={simrag_docs_count}")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues
    }


def get_system_metadata() -> Dict[str, Any]:
    """
    Get system metadata for experiment reproducibility
    
    Returns:
        Dictionary with system information:
        - python_version: Python version string
        - cuda_version: CUDA version if available
        - device_info: GPU/CPU information
    """
    import sys
    import platform
    
    metadata = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": platform.processor() if platform.system() != "Windows" else "N/A"
    }
    
    # CUDA version
    if TORCH_AVAILABLE and torch.cuda.is_available():
        metadata["cuda_version"] = torch.version.cuda
        metadata["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        metadata["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        metadata["gpu_count"] = torch.cuda.device_count()
    else:
        metadata["cuda_version"] = None
        metadata["gpu_name"] = None
        metadata["gpu_count"] = 0
    
    return metadata

