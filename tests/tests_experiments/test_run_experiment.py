"""
Tests for main experiment orchestrator
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from simrag_reproduction.experiments.run_experiment import run_complete_experiment


class TestRunCompleteExperiment:
    """Test run_complete_experiment function"""
    
    @patch('simrag_reproduction.experiments.run_experiment.run_baseline_test')
    @patch('simrag_reproduction.experiments.simrag.run_full_pipeline.run_full_pipeline')
    @patch('simrag_reproduction.experiments.run_experiment.compare_results')
    def test_run_full_pipeline(self, mock_compare, mock_simrag, mock_baseline):
        """Test running full experiment pipeline"""
        # Mock return values
        mock_baseline.return_value = {
            "_saved_filename": "baseline_results.json",
            "total_questions": 10,
            "average_context_score": 0.70
        }
        mock_simrag.return_value = {
            "_saved_filename": "simrag_results.json",
            "total_questions": 10,
            "average_context_score": 0.85
        }
        mock_compare.return_value = {
            "improvement": {
                "context_score_improvement_percent": 21.4
            }
        }
        
        # Run experiment
        run_complete_experiment(
            documents_folder="test_docs",
            use_real_datasets=False,
            skip_baseline=False,
            skip_simrag=False
        )
        
        # Verify all steps were called
        mock_baseline.assert_called_once()
        mock_simrag.assert_called_once()
        mock_compare.assert_called_once()
    
    @patch('simrag_reproduction.experiments.run_experiment.run_baseline_test')
    @patch('simrag_reproduction.experiments.simrag.run_full_pipeline.run_full_pipeline')
    @patch('simrag_reproduction.experiments.run_experiment.compare_results')
    def test_skip_baseline(self, mock_compare, mock_simrag, mock_baseline):
        """Test skipping baseline experiment"""
        mock_simrag.return_value = {
            "_saved_filename": "simrag_results.json"
        }
        mock_compare.return_value = {
            "improvement": {"context_score_improvement_percent": 10.0}
        }
        
        run_complete_experiment(
            documents_folder="test_docs",
            skip_baseline=True,
            skip_simrag=False,
            baseline_file="existing_baseline.json"
        )
        
        # Baseline should not be called
        mock_baseline.assert_not_called()
        # SimRAG and comparison should still be called
        mock_simrag.assert_called_once()
        mock_compare.assert_called_once()
    
    @patch('simrag_reproduction.experiments.run_experiment.run_baseline_test')
    @patch('simrag_reproduction.experiments.simrag.run_full_pipeline.run_full_pipeline')
    @patch('simrag_reproduction.experiments.run_experiment.compare_results')
    def test_skip_simrag(self, mock_compare, mock_simrag, mock_baseline):
        """Test skipping SimRAG experiment"""
        mock_baseline.return_value = {
            "_saved_filename": "baseline_results.json"
        }
        mock_compare.return_value = {
            "improvement": {"context_score_improvement_percent": 5.0}
        }
        
        run_complete_experiment(
            documents_folder="test_docs",
            skip_baseline=False,
            skip_simrag=True,
            simrag_file="existing_simrag.json"
        )
        
        # SimRAG should not be called
        mock_simrag.assert_not_called()
        # Baseline and comparison should still be called
        mock_baseline.assert_called_once()
        mock_compare.assert_called_once()
    
    @patch('simrag_reproduction.experiments.run_experiment.run_baseline_test')
    @patch('simrag_reproduction.experiments.simrag.run_full_pipeline.run_full_pipeline')
    @patch('simrag_reproduction.experiments.run_experiment.compare_results')
    def test_skip_both_experiments(self, mock_compare, mock_simrag, mock_baseline):
        """Test skipping both experiments (only comparison)"""
        mock_compare.return_value = {
            "improvement": {"context_score_improvement_percent": 15.0}
        }
        
        run_complete_experiment(
            documents_folder="test_docs",
            skip_baseline=True,
            skip_simrag=True,
            baseline_file="existing_baseline.json",
            simrag_file="existing_simrag.json"
        )
        
        # Neither experiment should be called
        mock_baseline.assert_not_called()
        mock_simrag.assert_not_called()
        # Only comparison should be called
        mock_compare.assert_called_once()
    
    @patch('simrag_reproduction.experiments.run_experiment.run_baseline_test')
    @patch('simrag_reproduction.experiments.simrag.run_full_pipeline.run_full_pipeline')
    @patch('simrag_reproduction.experiments.run_experiment.compare_results')
    def test_use_real_datasets_flag(self, mock_compare, mock_simrag, mock_baseline):
        """Test use_real_datasets flag is passed correctly"""
        mock_baseline.return_value = {"_saved_filename": "baseline.json"}
        mock_simrag.return_value = {"_saved_filename": "simrag.json"}
        mock_compare.return_value = {"improvement": {"context_score_improvement_percent": 10.0}}
        
        run_complete_experiment(
            documents_folder="test_docs",
            use_real_datasets=True,
            skip_baseline=False,
            skip_simrag=False
        )
        
        # Check that use_real_datasets was passed to SimRAG
        call_args = mock_simrag.call_args
        assert call_args[1]["use_real_datasets"] is True

