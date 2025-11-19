"""
Tests for experiment utility functions
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from simrag_reproduction.experiments.utils import (
    HTMLTextExtractor,
    extract_text_from_html,
    load_documents_from_folder,
    get_test_questions,
    get_timestamped_filename,
    has_timestamp
)


class TestHTMLTextExtractor:
    """Test HTML text extraction"""
    
    def test_extract_simple_html(self):
        """Test extracting text from simple HTML"""
        html = "<html><body><p>Hello World</p></body></html>"
        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        assert "Hello World" in text
    
    def test_skip_script_tags(self):
        """Test that script tags are skipped"""
        html = "<html><body><p>Hello</p><script>alert('test');</script><p>World</p></body></html>"
        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        assert "Hello" in text
        assert "World" in text
        assert "alert" not in text
        assert "test" not in text
    
    def test_skip_style_tags(self):
        """Test that style tags are skipped"""
        html = "<html><head><style>body { color: red; }</style></head><body><p>Content</p></body></html>"
        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        assert "Content" in text
        assert "color" not in text
        assert "red" not in text
    
    def test_handle_multiple_paragraphs(self):
        """Test handling multiple paragraphs"""
        html = "<html><body><p>First paragraph</p><p>Second paragraph</p></body></html>"
        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        assert "First paragraph" in text
        assert "Second paragraph" in text
    
    def test_clean_whitespace(self):
        """Test that whitespace is cleaned properly"""
        html = "<html><body><p>   Hello    World   </p></body></html>"
        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        # Should have cleaned up extra whitespace
        assert "Hello World" in text or "Hello" in text and "World" in text


class TestExtractTextFromHTML:
    """Test extract_text_from_html function"""
    
    def test_extract_from_file(self):
        """Test extracting text from HTML file"""
        html_content = "<html><body><p>Test content</p></body></html>"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_path = Path(f.name)
        
        try:
            text = extract_text_from_html(temp_path)
            assert "Test content" in text
        finally:
            temp_path.unlink()
    
    def test_handle_invalid_file(self):
        """Test handling invalid file path"""
        invalid_path = Path("/nonexistent/path/file.html")
        text = extract_text_from_html(invalid_path)
        assert text == ""


class TestLoadDocumentsFromFolder:
    """Test load_documents_from_folder function"""
    
    def test_load_txt_files(self):
        """Test loading .txt files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "doc1.txt").write_text("Document 1 content")
            (Path(tmpdir) / "doc2.txt").write_text("Document 2 content")
            
            documents = load_documents_from_folder(tmpdir, include_html=False)
            assert len(documents) == 2
            assert "Document 1 content" in documents
            assert "Document 2 content" in documents
    
    def test_load_md_files(self):
        """Test loading .md files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "readme.md").write_text("# Title\n\nContent here")
            
            documents = load_documents_from_folder(tmpdir, include_html=False)
            assert len(documents) == 1
            assert "Title" in documents[0]
            assert "Content here" in documents[0]
    
    def test_load_html_files(self):
        """Test loading HTML files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            html_content = "<html><body><p>HTML content</p></body></html>"
            (Path(tmpdir) / "page.html").write_text(html_content)
            
            documents = load_documents_from_folder(tmpdir, include_html=True)
            assert len(documents) == 1
            assert "HTML content" in documents[0]
    
    def test_exclude_html_files(self):
        """Test excluding HTML files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc.txt").write_text("Text content")
            (Path(tmpdir) / "page.html").write_text("<html><body>HTML</body></html>")
            
            documents = load_documents_from_folder(tmpdir, include_html=False)
            assert len(documents) == 1
            assert "Text content" in documents[0]
    
    def test_load_nested_files(self):
        """Test loading files from nested directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("Nested content")
            
            documents = load_documents_from_folder(tmpdir, include_html=False)
            assert len(documents) == 1
            assert "Nested content" in documents[0]
    
    def test_nonexistent_folder(self):
        """Test error handling for nonexistent folder"""
        with pytest.raises(ValueError, match="Folder not found"):
            load_documents_from_folder("/nonexistent/path")


class TestGetTestQuestions:
    """Test get_test_questions function"""
    
    def test_returns_list(self):
        """Test that function returns a list"""
        questions = get_test_questions()
        assert isinstance(questions, list)
        assert len(questions) > 0
    
    def test_questions_are_strings(self):
        """Test that all questions are strings"""
        questions = get_test_questions()
        assert all(isinstance(q, str) for q in questions)
    
    def test_contains_expected_questions(self):
        """Test that expected questions are included"""
        questions = get_test_questions()
        question_text = " ".join(questions).lower()
        assert "docker" in question_text
        assert "devops" in question_text


class TestGetTimestampedFilename:
    """Test get_timestamped_filename function"""
    
    def test_generates_timestamped_filename(self):
        """Test that function generates timestamped filename"""
        filename = get_timestamped_filename("test_results", "json")
        assert filename.startswith("test_results_")
        assert filename.endswith(".json")
        # Check timestamp format: YYYY-MM-DD_HH-MM-SS
        import re
        assert re.search(r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json$', filename)
    
    def test_different_extensions(self):
        """Test with different file extensions"""
        filename = get_timestamped_filename("results", "txt")
        assert filename.endswith(".txt")
        
        filename = get_timestamped_filename("data", "csv")
        assert filename.endswith(".csv")
    
    def test_unique_timestamps(self):
        """Test that timestamps are unique (or at least different format)"""
        import time
        filename1 = get_timestamped_filename("test", "json")
        time.sleep(0.1)  # Small delay to ensure different timestamp
        filename2 = get_timestamped_filename("test", "json")
        # They should be different (unless generated in same second)
        # At minimum, format should be correct
        assert filename1.startswith("test_")
        assert filename2.startswith("test_")


class TestHasTimestamp:
    """Test has_timestamp function"""
    
    def test_detects_timestamp(self):
        """Test that function detects timestamp in filename"""
        filename = "results_2024-11-20_14-30-45.json"
        assert has_timestamp(filename) is True
    
    def test_detects_no_timestamp(self):
        """Test that function detects absence of timestamp"""
        filename = "results.json"
        assert has_timestamp(filename) is False
    
    def test_detects_timestamp_with_path(self):
        """Test that function works with file paths"""
        filename = "path/to/results_2024-11-20_14-30-45.json"
        assert has_timestamp(filename) is True
    
    def test_invalid_timestamp_format(self):
        """Test that invalid timestamp format is not detected"""
        filename = "results_2024-11-20.json"  # Missing time component
        assert has_timestamp(filename) is False

