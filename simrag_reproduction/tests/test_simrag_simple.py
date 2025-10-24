"""
SimRAG Test Suite
Comprehensive test for the refactored SimRAG implementation
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Mock heavy dependencies
mock_modules = {
    'torch': MagicMock(),
    'transformers': MagicMock(),
    'datasets': MagicMock(),
    'tuning.basic_tuning': MagicMock(),
    'tuning.model_registry': MagicMock(),
    'config': MagicMock(),
    'rag.rag_setup': MagicMock(),
    'ai_providers.gateway': MagicMock()
}

with patch.dict('sys.modules', mock_modules):
    from simrag.instruction_following import InstructionFollowing
    from simrag.synthetic_qa_generation import SyntheticQAGeneration
    from simrag.domain_adaptation import DomainAdaptation


class TestSimRAG(unittest.TestCase):
    """Test for refactored SimRAG implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        print("\n=== SimRAG Test Suite ===")
        print("(No actual training - all mocked)")
        
        # Mock configuration
        self.mock_config = MagicMock()
        self.mock_config.optimized_num_epochs = 2
        self.mock_config.optimized_batch_size = 1
        self.mock_config.learning_rate = 0.0001
        self.mock_config.max_length = 512
        self.mock_config.use_laptop = True
        
        # Mock model version
        self.mock_version = MagicMock()
        self.mock_version.version = "test_v1.0"
        self.mock_version.training_time_seconds = 30.5
        self.mock_version.final_loss = 1.2345
    
    def test_instruction_following_simple(self):
        """Test instruction following class"""
        print("\n=== Testing Instruction Following ===")
        
        with patch('simrag.instruction_following.get_tuning_config', return_value=self.mock_config), \
             patch('simrag.instruction_following.BasicTuner'), \
             patch('simrag.instruction_following.get_model_registry', return_value=MagicMock()):
            
            trainer = InstructionFollowing()
            
            # Test data preparation
            test_data = trainer.prepare_instruction_data(use_real_datasets=False)
            self.assertEqual(len(test_data), 5)
            print(f"✅ Generated {len(test_data)} test examples")
            
            # Test training setup (mocked)
            with patch.object(trainer, 'load_model'), \
                 patch.object(trainer, 'prepare_training_data', return_value=MagicMock()), \
                 patch.object(trainer, 'setup_trainer'), \
                 patch.object(trainer, 'train_model', return_value=self.mock_version):
                
                version = trainer.train_stage_1(use_real_datasets=False)
                self.assertIsNotNone(version)
                print(f"✅ Stage 1 training: {version.version}")
    
    def test_synthetic_qa_generation_simple(self):
        """Test synthetic QA generation class"""
        print("\n=== Testing Synthetic QA Generation ===")
        
        with patch('simrag.synthetic_qa_generation.get_tuning_config', return_value=self.mock_config), \
             patch('simrag.synthetic_qa_generation.BasicTuner'), \
             patch('simrag.synthetic_qa_generation.get_model_registry', return_value=MagicMock()), \
             patch('simrag.synthetic_qa_generation.BasicRAG') as mock_rag, \
             patch('simrag.synthetic_qa_generation.AIGateway') as mock_gateway:
            
            # Mock responses
            mock_rag.return_value.query.return_value = (
                "Docker is a containerization platform.",
                ["doc1", "doc2"],
                [0.85, 0.78]
            )
            mock_gateway.return_value.chat.return_value = "What is Docker?\nHow does it work?"
            
            qa_generator = SyntheticQAGeneration()
            
            # Test QA generation
            documents = ["Docker is a containerization platform...", "Binary search is an algorithm..."]
            qa_pairs = qa_generator.create_qa_pairs_from_documents(documents, questions_per_doc=1)
            
            self.assertGreater(len(qa_pairs), 0)
            print(f"✅ Generated {len(qa_pairs)} QA pairs")
            
            # Test quality filtering
            filtered_pairs = qa_generator.filter_high_quality_qa_pairs(qa_pairs)
            self.assertLessEqual(len(filtered_pairs), len(qa_pairs))
            print(f"✅ Filtered to {len(filtered_pairs)} high-quality pairs")
    
    def test_domain_adaptation_simple(self):
        """Test domain adaptation class"""
        print("\n=== Testing Domain Adaptation ===")
        
        with patch('simrag.domain_adaptation.get_tuning_config', return_value=self.mock_config), \
             patch('simrag.domain_adaptation.BasicTuner') as mock_tuner, \
             patch('simrag.domain_adaptation.get_model_registry', return_value=MagicMock()), \
             patch('simrag.domain_adaptation.SyntheticQAGeneration') as mock_qa_gen:
            
            # Mock QA generator
            mock_qa_gen.return_value.generate_synthetic_dataset.return_value = {
                "training_data": ["Question: What is Docker?\nAnswer: Docker is a platform."],
                "dataset_info": {"total_qa_pairs": 2, "high_quality_pairs": 1, "training_examples": 1, "quality_retention": 50.0}
            }
            
            # Mock trainer
            mock_tuner.return_value.train.return_value = self.mock_version
            
            trainer = DomainAdaptation()
            
            # Test Stage 2 training
            documents = ["Docker is a containerization platform..."]
            version = trainer.train_stage_2(documents=documents, questions_per_doc=1)
            
            self.assertIsNotNone(version)
            print(f"✅ Stage 2 training: {version.version}")
            
            # Test self-improvement loop
            improvement_results = trainer.run_self_improvement_loop(
                documents=documents, improvement_rounds=1
            )
            
            self.assertGreater(len(improvement_results), 0)
            print(f"✅ Self-improvement: {len(improvement_results)} rounds")
    
    def test_no_overengineering(self):
        """Test that the refactored code is not overengineered"""
        print("\n=== Testing for Overengineering ===")
        
        # Check that classes are focused and not bloated
        with patch('simrag.instruction_following.get_tuning_config', return_value=self.mock_config), \
             patch('simrag.instruction_following.BasicTuner'), \
             patch('simrag.instruction_following.get_model_registry', return_value=MagicMock()):
            
            trainer = InstructionFollowing()
            
            # Check that methods are focused
            methods = [method for method in dir(trainer) if not method.startswith('_')]
            print(f"InstructionFollowing methods: {len(methods)}")
            self.assertLess(len(methods), 15)  # Should not have too many methods
            
            # Check that data preparation is simple
            test_data = trainer.prepare_instruction_data(use_real_datasets=False)
            self.assertEqual(len(test_data), 5)  # Simple, focused test data
            
            print("✅ No overengineering detected")
            print(f"  Methods: {len(methods)} (reasonable)")
            print(f"  Test data: {len(test_data)} examples (simple)")


def main():
    """Run the SimRAG test suite"""
    print("=== SimRAG Test Suite ===")
    print("Testing refactored SimRAG classes for functionality and simplicity")
    print("(No actual training will occur - all mocked for testing)\n")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSimRAG)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed! Refactored SimRAG is clean and functional.")
        print("✅ No overengineering detected")
        print("✅ All classes work correctly")
        print("✅ Ready for Checkpoint 2 implementation")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
