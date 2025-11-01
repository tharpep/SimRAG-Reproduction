"""
Simple tests for BasicTuner - focus on setup and configuration
"""

import pytest
from unittest.mock import patch, MagicMock
from tuning.basic_tuning import BasicTuner
from config import get_tuning_config


class TestBasicTuner:
    """Simple tests for BasicTuner core functionality"""
    
    def test_init(self):
        """Test basic initialization"""
        tuner = BasicTuner()
        assert tuner.model_name == "qwen3:1.7b"
        assert tuner.device in ["cpu", "cuda", "mps"]
        assert tuner.tokenizer is None
        assert tuner.model is None
        assert tuner.config is None
        assert tuner.registry is None
    
    def test_init_custom(self):
        """Test initialization with custom parameters"""
        tuner = BasicTuner(model_name="llama3.2:1b", device="cpu")
        assert tuner.model_name == "llama3.2:1b"
        assert tuner.device == "cpu"
        assert tuner.config is None
        assert tuner.registry is None
    
    @patch('tuning.basic_tuning.AutoTokenizer')
    @patch('tuning.basic_tuning.AutoModelForCausalLM')
    def test_model_name_mapping(self, mock_model, mock_tokenizer):
        """Test that model names map to correct Hugging Face models"""
        # Test llama3.2:1b mapping
        tuner = BasicTuner(model_name="llama3.2:1b")
        
        # Mock the tokenizer and model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        tuner.load_model()
        
        # Verify it called with the correct HF model
        mock_tokenizer.from_pretrained.assert_called_with("meta-llama/Llama-3.2-1B")
        # Check that the model was called with the correct model name (ignore other parameters)
        assert mock_model.from_pretrained.call_args[0][0] == "meta-llama/Llama-3.2-1B"
    
    @patch('tuning.basic_tuning.Dataset')
    def test_prepare_data_structure(self, mock_dataset_class):
        """Test data preparation creates correct structure"""
        tuner = BasicTuner()
        
        # Mock the dataset and tokenizer
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = MagicMock()
        mock_dataset_class.from_dict.return_value = mock_dataset
        
        tuner.tokenizer = MagicMock()
        tuner.tokenizer.return_value = {
            'input_ids': [[1, 2, 3], [4, 5, 6]],  # Correct format for 2 texts
            'attention_mask': [[1, 1, 1], [1, 1, 1]]
        }
        
        texts = ["Hello world", "Test sentence"]
        dataset = tuner.prepare_data(texts)
        
        assert dataset is not None
        mock_dataset_class.from_dict.assert_called_once()
        mock_dataset.map.assert_called_once()
    
    def test_setup_trainer_parameters(self):
        """Test trainer setup with correct parameters"""
        tuner = BasicTuner()
        tuner.model = MagicMock()
        tuner.tokenizer = MagicMock()
        
        mock_dataset = MagicMock()
        
        with patch('tuning.basic_tuning.TrainingArguments') as mock_args, \
             patch('tuning.basic_tuning.DataCollatorForLanguageModeling'), \
             patch('tuning.basic_tuning.Trainer') as mock_trainer:
            
            tuner.setup_trainer(
                train_dataset=mock_dataset,
                output_dir="./test_output",
                num_epochs=2,
                batch_size=4,
                learning_rate=1e-4
            )
            
            # Verify TrainingArguments was called with correct parameters
            mock_args.assert_called_once()
            args_call = mock_args.call_args[1]  # Get keyword arguments
            assert args_call['output_dir'] == "./test_output"
            assert args_call['num_train_epochs'] == 2
            assert args_call['per_device_train_batch_size'] == 4
            assert args_call['learning_rate'] == 1e-4
            
            assert tuner.trainer is not None
    
    def test_error_handling_no_trainer(self):
        """Test that methods fail gracefully without proper setup"""
        tuner = BasicTuner()
        
        with pytest.raises(ValueError, match="Trainer not setup"):
            tuner.train()
        
        with pytest.raises(ValueError, match="No trainer available"):
            tuner.save_model()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            tuner.generate_text("Test prompt")
    
    def test_model_info_structure(self):
        """Test model info returns correct structure"""
        tuner = BasicTuner()
        
        # Test without model
        info = tuner.get_model_info()
        assert info == {"error": "Model not loaded"}
        
        # Test with mock model
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 1000
        mock_param.element_size.return_value = 4
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        tuner.model = mock_model
        
        info = tuner.get_model_info()
        assert 'model_name' in info
        assert 'device' in info
        assert 'num_parameters' in info
        assert 'trainable_parameters' in info
        assert 'model_size_mb' in info


class TestTuningConfig:
    """Test tuning configuration"""
    
    def test_config_hardware_optimization(self):
        """Test that config provides hardware-optimized settings"""
        config = get_tuning_config()
        
        # Test laptop optimization
        if config.use_laptop:
            assert config.optimized_batch_size == 1
            assert config.optimized_num_epochs == 1
        else:
            assert config.optimized_batch_size == 8
            assert config.optimized_num_epochs == 3
    
    def test_model_name_mapping(self):
        """Test config provides correct model names"""
        config = get_tuning_config()
        
        if config.use_laptop:
            assert config.model_name == "llama3.2:1b"
            assert "1b" in config.output_dir
        else:
            assert config.model_name == "qwen3:8b"
            assert "8b" in config.output_dir


class TestTuningIntegration:
    """Test tuning workflow integration"""
    
    @patch('tuning.basic_tuning.Dataset')
    @patch('tuning.basic_tuning.AutoTokenizer')
    @patch('tuning.basic_tuning.AutoModelForCausalLM')
    def test_full_workflow_setup(self, mock_model, mock_tokenizer, mock_dataset_class):
        """Test that the full workflow can be set up (without actual training)"""
        config = get_tuning_config()
        
        # Initialize tuner
        tuner = BasicTuner(
            model_name=config.model_name,
            device="cpu"  # Use CPU for testing
        )
        
        # Mock the tokenizer and model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        # Test that we can prepare sample data
        sample_texts = [
            "This is a test sentence for tuning.",
            "Another example for the training data."
        ]
        
        # Mock the dataset
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = MagicMock()
        mock_dataset_class.from_dict.return_value = mock_dataset
        
        tuner.tokenizer = MagicMock()
        tuner.tokenizer.return_value = {
            'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],  # Correct format for 2 texts
            'attention_mask': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        }
        
        dataset = tuner.prepare_data(sample_texts)
        assert dataset is not None
        
        # Test that we can set up trainer (without actually training)
        tuner.model = MagicMock()
        
        with patch('tuning.basic_tuning.TrainingArguments'), \
             patch('tuning.basic_tuning.DataCollatorForLanguageModeling'), \
             patch('tuning.basic_tuning.Trainer'):
            
            tuner.setup_trainer(
                train_dataset=dataset,
                output_dir=config.output_dir,
                num_epochs=config.optimized_num_epochs,
                batch_size=config.optimized_batch_size,
                learning_rate=config.learning_rate
            )
            
            assert tuner.trainer is not None
        
        print(f"[OK] Tuning workflow setup successful!")
        print(f"   Model: {config.model_name}")
        print(f"   Batch size: {config.optimized_batch_size}")
        print(f"   Epochs: {config.optimized_num_epochs}")
        print(f"   Output: {config.output_dir}")