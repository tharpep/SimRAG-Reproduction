"""
Tests for comparison module
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from simrag_reproduction.experiments.comparison.compare_results import (
    load_results_file,
    compare_results
)


class TestLoadResultsFile:
    """Test load_results_file function"""
    
    def test_load_valid_json(self):
        """Test loading valid JSON file"""
        data = {
            "test": "data",
            "results": [1, 2, 3],
            "metrics": {"score": 0.85}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)
        
        try:
            loaded = load_results_file(str(temp_path))
            assert loaded == data
            assert loaded["test"] == "data"
            assert loaded["results"] == [1, 2, 3]
            assert loaded["metrics"]["score"] == 0.85
        finally:
            temp_path.unlink()
    
    def test_nonexistent_file(self):
        """Test error handling for nonexistent file"""
        with pytest.raises(FileNotFoundError, match="Results file not found"):
            load_results_file("/nonexistent/path/results.json")
    
    def test_invalid_json(self):
        """Test error handling for invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_results_file(str(temp_path))
        finally:
            temp_path.unlink()


class TestCompareResults:
    """Test compare_results function"""
    
    def test_compare_basic_results(self):
        """Test basic comparison of results"""
        baseline_data = {
            "total_questions": 10,
            "correct_answers": 6,
            "summary": {
                "avg_context_score": 0.70,
                "avg_response_time": 1.5
            },
            "questions": ["Q1", "Q2"]
        }
        
        simrag_data = {
            "total_questions": 10,
            "correct_answers": 8,
            "testing": {
                "avg_context_score": 0.85,
                "avg_response_time": 1.2
            },
            "questions": ["Q1", "Q2"],
            "stage1": {"version": "v1.0"},
            "stage2": {"version": "v1.0"}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.json"
            simrag_file = Path(tmpdir) / "simrag.json"
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f)
            with open(simrag_file, 'w') as f:
                json.dump(simrag_data, f)
            
            comparison = compare_results(
                baseline_file=str(baseline_file),
                simrag_file=str(simrag_file),
                output_file=None,
                use_timestamp=False
            )
            
            # Check comparison structure
            assert "baseline" in comparison
            assert "simrag" in comparison
            assert "improvement" in comparison
            
            # Check improvement calculations
            assert comparison["improvement"]["context_score_improvement"] > 0
            assert comparison["improvement"]["context_score_improvement_percent"] > 0
    
    def test_compare_with_missing_fields(self):
        """Test comparison with missing optional fields"""
        baseline_data = {
            "total_questions": 5,
            "summary": {
                "avg_context_score": 0.60
            },
            "questions": []
        }
        
        simrag_data = {
            "total_questions": 5,
            "testing": {
                "avg_context_score": 0.80
            },
            "questions": []
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.json"
            simrag_file = Path(tmpdir) / "simrag.json"
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f)
            with open(simrag_file, 'w') as f:
                json.dump(simrag_data, f)
            
            comparison = compare_results(
                baseline_file=str(baseline_file),
                simrag_file=str(simrag_file),
                output_file=None,
                use_timestamp=False
            )
            
            # Should still work with missing fields
            assert "improvement" in comparison
            assert comparison["improvement"]["context_score_improvement"] > 0
    
    def test_saves_output_file(self):
        """Test that comparison saves output file when specified"""
        baseline_data = {
            "summary": {"avg_context_score": 0.70},
            "questions": []
        }
        simrag_data = {
            "testing": {"avg_context_score": 0.85},
            "questions": []
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.json"
            simrag_file = Path(tmpdir) / "simrag.json"
            output_file = Path(tmpdir) / "comparison.json"
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f)
            with open(simrag_file, 'w') as f:
                json.dump(simrag_data, f)
            
            comparison = compare_results(
                baseline_file=str(baseline_file),
                simrag_file=str(simrag_file),
                output_file=str(output_file),
                use_timestamp=False
            )
            
            # Check that file was created
            assert output_file.exists()
            
            # Check that file contains comparison data
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            # Remove _saved_filename for comparison since it's added by the function
            comparison_without_saved = {k: v for k, v in comparison.items() if k != "_saved_filename"}
            saved_without_saved = {k: v for k, v in saved_data.items() if k != "_saved_filename"}
            assert saved_without_saved == comparison_without_saved
    
    def test_handles_negative_improvement(self):
        """Test handling of negative improvement (regression)"""
        baseline_data = {
            "summary": {"avg_context_score": 0.85},
            "questions": []
        }
        simrag_data = {
            "testing": {"avg_context_score": 0.70},
            "questions": []
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.json"
            simrag_file = Path(tmpdir) / "simrag.json"
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f)
            with open(simrag_file, 'w') as f:
                json.dump(simrag_data, f)
            
            comparison = compare_results(
                baseline_file=str(baseline_file),
                simrag_file=str(simrag_file),
                output_file=None,
                use_timestamp=False
            )
            
            # Should handle negative improvement
            assert comparison["improvement"]["context_score_improvement"] < 0
            assert comparison["improvement"]["context_score_improvement_percent"] < 0

