"""
SimRAG Test Suite
Tests for SimRAG three-stage pipeline using pytest
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Imports will be done within patched contexts to handle dependencies


class TestSimRAG:
    """Test suite for SimRAG implementation"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = MagicMock()
        config.optimized_num_epochs = 2
        config.optimized_batch_size = 1
        config.learning_rate = 0.0001
        config.max_length = 512
        config.model_size = "small"
        config.simrag_stage_1_epochs = 1
        config.simrag_stage_2_epochs = 1
        config.simrag_improvement_rounds = 2
        config.simrag_questions_per_doc = 2
        config.simrag_min_context_score = 0.7
        config.model_registry_path = "./tuned_models/llama_1b/model_registry.json"
        return config
    
    @pytest.fixture
    def mock_version(self):
        """Mock model version for testing"""
        version = MagicMock()
        version.version = "test_v1.0"
        version.training_time_seconds = 30.5
        version.final_loss = 1.2345
        return version
    
    def test_instruction_following_initialization(self, mock_config):
        """Test InstructionFollowing class initialization"""
        print("\n=== Testing Instruction Following Initialization ===")
        
        with patch('simrag_reproduction.config.get_tuning_config', return_value=mock_config), \
             patch('simrag_reproduction.tuning.basic_tuning.BasicTuner'), \
             patch('simrag_reproduction.tuning.model_registry.get_model_registry', return_value=MagicMock()):
            
            from simrag_reproduction.simrag.instruction_following import InstructionFollowing
            
            trainer = InstructionFollowing()
            assert trainer is not None
            assert trainer.model_name == "Qwen/Qwen2.5-1.5B-Instruct"  # Updated default
            print("[OK] InstructionFollowing initialized successfully")
    
    def test_instruction_following_data_preparation(self, mock_config):
        """Test instruction following data preparation"""
        print("\n=== Testing Instruction Following Data Preparation ===")
        
        with patch('simrag_reproduction.config.get_tuning_config', return_value=mock_config), \
             patch('simrag_reproduction.tuning.basic_tuning.BasicTuner'), \
             patch('simrag_reproduction.tuning.model_registry.get_model_registry', return_value=MagicMock()):
            
            from simrag_reproduction.simrag.instruction_following import InstructionFollowing
            
            trainer = InstructionFollowing()
            test_data = trainer.prepare_instruction_data(use_real_datasets=False)
            
            assert len(test_data) == 5
            assert all("Question:" in item and "Answer:" in item for item in test_data)
            print(f"[OK] Generated {len(test_data)} test instruction examples")
    
    def test_synthetic_qa_generation_initialization(self, mock_config):
        """Test SyntheticQAGeneration class initialization"""
        print("\n=== Testing Synthetic QA Generation Initialization ===")
        
        with patch('simrag_reproduction.config.get_tuning_config', return_value=mock_config), \
             patch('simrag_reproduction.tuning.basic_tuning.BasicTuner'), \
             patch('simrag_reproduction.tuning.model_registry.get_model_registry', return_value=MagicMock()), \
             patch('simrag_reproduction.rag.rag_setup.BasicRAG'), \
             patch('simrag_reproduction.ai_providers.gateway.AIGateway'):
            
            from simrag_reproduction.simrag.synthetic_qa_generation import SyntheticQAGeneration
            
            qa_generator = SyntheticQAGeneration()
            assert qa_generator is not None
            assert qa_generator.model_name == "Qwen/Qwen2.5-1.5B-Instruct"  # Updated default
            print("[OK] SyntheticQAGeneration initialized successfully")
    
    def test_synthetic_qa_generation_question_parsing(self, mock_config):
        """Test question parsing from LLM response"""
        print("\n=== Testing Question Parsing ===")
        
        with patch('simrag_reproduction.config.get_tuning_config', return_value=mock_config), \
             patch('simrag_reproduction.tuning.basic_tuning.BasicTuner'), \
             patch('simrag_reproduction.tuning.model_registry.get_model_registry', return_value=MagicMock()), \
             patch('simrag_reproduction.rag.rag_setup.BasicRAG'), \
             patch('simrag_reproduction.ai_providers.gateway.AIGateway'):
            
            from simrag_reproduction.simrag.synthetic_qa_generation import SyntheticQAGeneration
            
            qa_generator = SyntheticQAGeneration()
            
            # Test various response formats
            response1 = "What is Docker?\nHow does it work?\nExplain containers."
            questions1 = qa_generator._parse_questions(response1)
            assert len(questions1) > 0
            
            response2 = "1. What is Python?\n2. How to use it?"
            questions2 = qa_generator._parse_questions(response2)
            assert len(questions2) > 0
            
            print(f"[OK] Question parsing works for different formats")
    
    def test_synthetic_qa_generation_filtering(self, mock_config):
        """Test QA pair quality filtering"""
        print("\n=== Testing QA Pair Filtering ===")
        
        with patch('simrag_reproduction.config.get_tuning_config', return_value=mock_config), \
             patch('simrag_reproduction.tuning.basic_tuning.BasicTuner'), \
             patch('simrag_reproduction.tuning.model_registry.get_model_registry', return_value=MagicMock()), \
             patch('simrag_reproduction.rag.rag_setup.BasicRAG'), \
             patch('simrag_reproduction.ai_providers.gateway.AIGateway'):
            
            from simrag_reproduction.simrag.synthetic_qa_generation import SyntheticQAGeneration
            
            qa_generator = SyntheticQAGeneration()
            
            # Test data with mixed quality
            qa_pairs = [
                {
                    "question": "What is Docker?",
                    "answer": "Docker is a containerization platform that enables applications to run in isolated environments.",
                    "context_scores": [0.85, 0.78]
                },
                {
                    "question": "Test question?",
                    "answer": "Short",  # Too short
                    "context_scores": [0.65]  # Low score
                },
                {
                    "question": "Another question?",
                    "answer": "No relevant documents found",  # Invalid answer
                    "context_scores": [0.80]
                }
            ]
            
            filtered = qa_generator.filter_high_quality_qa_pairs(qa_pairs, min_context_score=0.7)
            
            assert len(filtered) <= len(qa_pairs)
            assert len(filtered) == 1  # Only the first one should pass
            print(f"[OK] Filtered {len(qa_pairs)} pairs to {len(filtered)} high-quality pairs")
    
    def test_synthetic_qa_generation_empty_input(self, mock_config):
        """Test handling of empty input"""
        print("\n=== Testing Empty Input Handling ===")
        
        with patch('simrag_reproduction.config.get_tuning_config', return_value=mock_config), \
             patch('simrag_reproduction.tuning.basic_tuning.BasicTuner'), \
             patch('simrag_reproduction.tuning.model_registry.get_model_registry', return_value=MagicMock()), \
             patch('simrag_reproduction.rag.rag_setup.BasicRAG'), \
             patch('simrag_reproduction.ai_providers.gateway.AIGateway'):
            
            from simrag_reproduction.simrag.synthetic_qa_generation import SyntheticQAGeneration
            
            qa_generator = SyntheticQAGeneration()
            
            # Test with empty list
            empty_pairs = []
            filtered = qa_generator.filter_high_quality_qa_pairs(empty_pairs)
            assert len(filtered) == 0
            print("[OK] Empty input handled correctly")
    
    def test_domain_adaptation_initialization(self, mock_config):
        """Test DomainAdaptation class initialization"""
        print("\n=== Testing Domain Adaptation Initialization ===")
        
        with patch('simrag_reproduction.config.get_tuning_config', return_value=mock_config), \
             patch('simrag_reproduction.tuning.basic_tuning.BasicTuner'), \
             patch('simrag_reproduction.tuning.model_registry.get_model_registry', return_value=MagicMock()), \
             patch('simrag_reproduction.simrag.synthetic_qa_generation.SyntheticQAGeneration'):
            
            from simrag_reproduction.simrag.domain_adaptation import DomainAdaptation
            
            trainer = DomainAdaptation()
            assert trainer is not None
            assert trainer.model_name == "Qwen/Qwen2.5-1.5B-Instruct"  # Updated default
            print("[OK] DomainAdaptation initialized successfully")
    
    def test_domain_adaptation_with_stage_1_model(self, mock_config):
        """Test DomainAdaptation with Stage 1 model path"""
        print("\n=== Testing Domain Adaptation with Stage 1 Model ===")
        
        with patch('simrag_reproduction.config.get_tuning_config', return_value=mock_config), \
             patch('simrag_reproduction.tuning.basic_tuning.BasicTuner'), \
             patch('simrag_reproduction.tuning.model_registry.get_model_registry', return_value=MagicMock()), \
             patch('simrag_reproduction.simrag.synthetic_qa_generation.SyntheticQAGeneration'):
            
            from simrag_reproduction.simrag.domain_adaptation import DomainAdaptation
            
            trainer = DomainAdaptation(stage_1_model_path="./tuned_models/llama_1b/v1.0")
            assert trainer.stage_1_model_path == "./tuned_models/llama_1b/v1.0"
            print("[OK] DomainAdaptation accepts Stage 1 model path")
    
    def test_simrag_base_functionality(self, mock_config):
        """Test SimRAGBase common functionality"""
        print("\n=== Testing SimRAG Base Functionality ===")
        
        with patch('simrag_reproduction.config.get_tuning_config', return_value=mock_config), \
             patch('simrag_reproduction.tuning.basic_tuning.BasicTuner'), \
             patch('simrag_reproduction.tuning.model_registry.get_model_registry', return_value=MagicMock()):
            
            from simrag_reproduction.simrag.instruction_following import InstructionFollowing
            
            trainer = InstructionFollowing()
            
            # Test that base methods exist and are callable
            assert hasattr(trainer, 'load_model')
            assert hasattr(trainer, 'prepare_training_data')
            assert hasattr(trainer, 'setup_trainer')
            assert hasattr(trainer, 'train_model')
            assert hasattr(trainer, 'test_performance')
            assert hasattr(trainer, 'calculate_improvement_metrics')
            
            print("[OK] All SimRAGBase methods available")

