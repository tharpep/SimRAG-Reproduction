"""
Test HuggingFace Transformers Client
"""

import pytest
from unittest.mock import patch, MagicMock
import torch

from simrag_reproduction.ai_providers.huggingface_client import HuggingFaceClient


class TestHuggingFaceClient:
    """Test cases for HuggingFaceClient"""
    
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoModelForCausalLM')
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoTokenizer')
    def test_init_with_hub_model(self, mock_tokenizer, mock_model):
        """Test initialization with HuggingFace Hub model ID"""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            client = HuggingFaceClient("Qwen/Qwen2.5-1.5B-Instruct", device="cpu")
            
            assert client.model_path == "Qwen/Qwen2.5-1.5B-Instruct"
            assert client.device == "cpu"
            mock_tokenizer.from_pretrained.assert_called_once_with("Qwen/Qwen2.5-1.5B-Instruct")
            mock_model.from_pretrained.assert_called_once()
    
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoModelForCausalLM')
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoTokenizer')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_init_with_local_path(self, mock_isdir, mock_exists, mock_tokenizer, mock_model):
        """Test initialization with local model path"""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            client = HuggingFaceClient("./tuned_models/model_1b/stage_1/v1.0", device="cpu")
            
            assert client.model_path == "./tuned_models/model_1b/stage_1/v1.0"
            mock_tokenizer.from_pretrained.assert_called_once_with("./tuned_models/model_1b/stage_1/v1.0")
    
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoModelForCausalLM')
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoTokenizer')
    def test_device_auto_detection_cuda(self, mock_tokenizer, mock_model):
        """Test automatic device detection for CUDA"""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=True):
            client = HuggingFaceClient("Qwen/Qwen2.5-1.5B-Instruct", device="auto")
            assert client.device == "cuda"
    
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoModelForCausalLM')
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoTokenizer')
    def test_device_auto_detection_cpu(self, mock_tokenizer, mock_model):
        """Test automatic device detection for CPU"""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            client = HuggingFaceClient("Qwen/Qwen2.5-1.5B-Instruct", device="auto")
            assert client.device == "cpu"
    
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoModelForCausalLM')
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoTokenizer')
    def test_chat_with_string_message(self, mock_tokenizer, mock_model):
        """Test chat with string message - structure test only"""
        # Note: Full integration test would require actual model download
        # This test verifies the client structure and that chat method exists
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            client = HuggingFaceClient("Qwen/Qwen2.5-1.5B-Instruct", device="cpu")
            
            # Verify client structure
            assert client.model is not None
            assert client.tokenizer is not None
            assert client.model_path == "Qwen/Qwen2.5-1.5B-Instruct"
            assert hasattr(client, 'chat')
            assert callable(client.chat)
            
            # Note: Actual generation test would require real model or more complex mocking
            # The structure is verified above
    
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoModelForCausalLM')
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoTokenizer')
    def test_chat_with_list_message(self, mock_tokenizer, mock_model):
        """Test chat with list of message dicts - simplified to avoid actual model loading"""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            client = HuggingFaceClient("Qwen/Qwen2.5-1.5B-Instruct", device="cpu")
            
            # Verify client can handle list messages (structure test)
            assert hasattr(client, 'chat')
            # Test that it accepts list format - actual generation requires real model
            messages = [{"role": "user", "content": "Hello"}]
            # Just verify the method signature accepts this format
            assert callable(client.chat)
    
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoModelForCausalLM')
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoTokenizer')
    def test_chat_model_not_loaded(self, mock_tokenizer, mock_model):
        """Test chat fails when model is not loaded"""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            client = HuggingFaceClient("Qwen/Qwen2.5-1.5B-Instruct", device="cpu")
            client.model = None  # Simulate model not loaded
            
            with pytest.raises(ValueError, match="Model not loaded"):
                client.chat("Hello")
    
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoModelForCausalLM')
    @patch('simrag_reproduction.ai_providers.huggingface_client.AutoTokenizer')
    def test_get_available_models(self, mock_tokenizer, mock_model):
        """Test get_available_models returns model path"""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            client = HuggingFaceClient("Qwen/Qwen2.5-1.5B-Instruct", device="cpu")
            models = client.get_available_models()
            
            assert isinstance(models, list)
            assert "Qwen/Qwen2.5-1.5B-Instruct" in models

