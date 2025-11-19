"""
Utility functions for experiments
Helper functions for document loading, HTML extraction, etc.
"""

from pathlib import Path
from typing import List
import re
from html.parser import HTMLParser


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
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
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

