"""
Simple tests for tuning demo functionality
"""

import pytest
from unittest.mock import patch, MagicMock
from tuning.demo import run_tuning_demo


class TestTuningDemo:
    """Simple tests for tuning demo"""
    
    @patch('tuning.demo.BasicTuner')
    @patch('tuning.demo.get_tuning_config')
    def test_demo_workflow_setup(self, mock_get_config, mock_tuner_class):
        """Test that demo workflow can be set up correctly"""
        # Setup config mock
        mock_config = MagicMock()
        mock_config.model_name = "llama3.2:1b"
        mock_config.device = "cpu"
        mock_config.use_laptop = True
        mock_config.optimized_batch_size = 1
        mock_config.optimized_num_epochs = 1
        mock_config.max_length = 512
        mock_config.output_dir = "./test_output"
        mock_config.learning_rate = 5e-5
        mock_get_config.return_value = mock_config
        
        # Setup tuner mock
        mock_tuner = MagicMock()
        mock_tuner.get_model_info.return_value = {
            'model_name': 'llama3.2:1b', 
            'num_parameters': 1000000
        }
        mock_tuner.prepare_data.return_value = MagicMock()
        mock_tuner.generate_text.return_value = "Generated text response"
        
        # Mock the train method to return a version object
        mock_version = MagicMock()
        mock_version.version = "v1.0"
        mock_version.training_time_seconds = 45.2
        mock_version.final_loss = 2.3456
        mock_version.model_size_mb = 1.2
        mock_tuner.train.return_value = mock_version
        
        # Mock the registry
        mock_tuner.registry = MagicMock()
        mock_tuner.registry.list_versions.return_value = None
        
        mock_tuner_class.return_value = mock_tuner
        
        # Run demo
        run_tuning_demo(mode="quick")
        
        # Verify key steps were called
        mock_tuner.load_model.assert_called_once()
        mock_tuner.prepare_data.assert_called_once()
        mock_tuner.setup_trainer.assert_called_once()
        mock_tuner.train.assert_called_once()
        mock_tuner.save_model.assert_called_once()
        mock_tuner.generate_text.assert_called()
        
        # Verify setup_trainer was called with optimized parameters
        setup_call = mock_tuner.setup_trainer.call_args[1]
        assert setup_call['batch_size'] == 1  # Laptop optimized
        assert setup_call['num_epochs'] == 1  # Quick mode
    
    @patch('tuning.demo.BasicTuner')
    @patch('tuning.demo.get_tuning_config')
    def test_demo_uses_correct_training_data(self, mock_get_config, mock_tuner_class):
        """Test that demo uses correct training data for each mode"""
        # Setup config mock
        mock_config = MagicMock()
        mock_config.model_name = "llama3.2:1b"
        mock_config.device = "cpu"
        mock_config.use_laptop = True
        mock_config.optimized_batch_size = 1
        mock_config.optimized_num_epochs = 1
        mock_config.max_length = 512
        mock_config.output_dir = "./test_output"
        mock_config.learning_rate = 5e-5
        mock_get_config.return_value = mock_config
        
        # Setup tuner mock
        mock_tuner = MagicMock()
        mock_tuner.get_model_info.return_value = {
            'model_name': 'llama3.2:1b', 
            'num_parameters': 1000000
        }
        mock_tuner.prepare_data.return_value = MagicMock()
        mock_tuner.generate_text.return_value = "Generated text response"
        
        # Mock the train method to return a version object
        mock_version = MagicMock()
        mock_version.version = "v1.0"
        mock_version.training_time_seconds = 45.2
        mock_version.final_loss = 2.3456
        mock_version.model_size_mb = 1.2
        mock_tuner.train.return_value = mock_version
        
        # Mock the registry
        mock_tuner.registry = MagicMock()
        mock_tuner.registry.list_versions.return_value = None
        
        mock_tuner_class.return_value = mock_tuner
        
        # Test quick mode
        run_tuning_demo(mode="quick")
        prepare_call = mock_tuner.prepare_data.call_args[0][0]
        assert len(prepare_call) == 5  # Quick mode has 5 texts
        assert "Machine learning is fascinating." in prepare_call
        
        # Reset mock
        mock_tuner.reset_mock()
        
        # Test full mode
        run_tuning_demo(mode="full")
        prepare_call = mock_tuner.prepare_data.call_args[0][0]
        assert len(prepare_call) == 10  # Full mode has 10 texts
        # Check for a shorter, more likely text from the demo
        assert any("Machine learning" in text for text in prepare_call)
    
    @patch('tuning.demo.BasicTuner')
    @patch('tuning.demo.get_tuning_config')
    def test_demo_handles_errors_gracefully(self, mock_get_config, mock_tuner_class):
        """Test demo handles errors gracefully"""
        # Setup config mock
        mock_config = MagicMock()
        mock_config.model_name = "llama3.2:1b"
        mock_config.device = "cpu"
        mock_config.use_laptop = True
        mock_config.optimized_batch_size = 1
        mock_config.optimized_num_epochs = 1
        mock_config.max_length = 512
        mock_config.output_dir = "./test_output"
        mock_config.learning_rate = 5e-5
        mock_get_config.return_value = mock_config
        
        # Setup tuner mock with error
        mock_tuner = MagicMock()
        mock_tuner.get_model_info.return_value = {
            'model_name': 'llama3.2:1b', 
            'num_parameters': 1000000
        }
        mock_tuner.prepare_data.return_value = MagicMock()
        mock_tuner.generate_text.return_value = "Generated text response"
        mock_tuner.train.side_effect = Exception("Training failed")
        mock_tuner_class.return_value = mock_tuner
        
        # Test that demo catches the error and continues (demo catches exceptions)
        run_tuning_demo(mode="quick")  # Should not raise, just print error
        
        # Verify setup completed before error
        mock_tuner.load_model.assert_called_once()
        mock_tuner.prepare_data.assert_called_once()
        mock_tuner.setup_trainer.assert_called_once()
        mock_tuner.train.assert_called_once()
        
        # Verify save was not called due to error
        mock_tuner.save_model.assert_not_called()
    
    def test_demo_configuration_integration(self):
        """Test that demo integrates correctly with configuration"""
        from config import get_tuning_config
        
        config = get_tuning_config()
        
        # Verify config provides all required fields
        assert hasattr(config, 'model_name')
        assert hasattr(config, 'device')
        assert hasattr(config, 'optimized_batch_size')
        assert hasattr(config, 'optimized_num_epochs')
        assert hasattr(config, 'max_length')
        assert hasattr(config, 'output_dir')
        assert hasattr(config, 'learning_rate')
        
        # Verify hardware optimization
        if config.use_laptop:
            assert config.optimized_batch_size == 1
            assert config.optimized_num_epochs == 1
        else:
            assert config.optimized_batch_size == 8
            assert config.optimized_num_epochs == 3
        
        print(f"[OK] Demo configuration integration successful!")
        print(f"   Hardware: {'Laptop' if config.use_laptop else 'PC'}")
        print(f"   Model: {config.model_name}")
        print(f"   Optimized batch size: {config.optimized_batch_size}")
        print(f"   Optimized epochs: {config.optimized_num_epochs}")