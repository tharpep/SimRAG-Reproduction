"""
Test AI Gateway functionality
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from simrag_reproduction.ai_providers.gateway import AIGateway


class TestAIGateway:
    """Test cases for AIGateway"""
    
    def test_init_with_config(self):
        """Test gateway initialization with config"""
        config = {
            "purdue": {"api_key": "test-key"},
            "ollama": {"base_url": "http://localhost:11434", "default_model": "test-model"}
        }
        
        with patch('simrag_reproduction.ai_providers.gateway.PurdueGenAI') as mock_purdue, \
             patch('simrag_reproduction.ai_providers.gateway.OllamaClient') as mock_ollama:
            
            gateway = AIGateway(config)
            
            # Should initialize both providers
            mock_purdue.assert_called_once_with("test-key")
            mock_ollama.assert_called_once()
    
    def test_init_from_env_vars(self):
        """Test gateway initialization from environment variables"""
        with patch.dict(os.environ, {'PURDUE_API_KEY': 'test-key'}), \
             patch('simrag_reproduction.ai_providers.gateway.PurdueGenAI') as mock_purdue, \
             patch('simrag_reproduction.ai_providers.gateway.HuggingFaceClient') as mock_hf:
            
            gateway = AIGateway()
            # HuggingFace should be initialized by default
            mock_hf.assert_called_once()
            # Purdue should also be initialized if API key is present
            mock_purdue.assert_called_once()
    
    def test_init_with_ollama_env(self):
        """Test gateway initialization with Ollama environment variable"""
        with patch.dict(os.environ, {'USE_OLLAMA': 'true', 'MODEL_SIZE': 'small'}), \
             patch('simrag_reproduction.ai_providers.gateway.OllamaClient') as mock_ollama:
            
            gateway = AIGateway()
            mock_ollama.assert_called_once()
    
    def test_get_available_providers(self):
        """Test getting available providers"""
        config = {"purdue": {"api_key": "test-key"}}
        
        with patch('simrag_reproduction.ai_providers.gateway.PurdueGenAI'):
            gateway = AIGateway(config)
            providers = gateway.get_available_providers()
            assert "purdue" in providers
    
    def test_chat_with_provider_selection(self):
        """Test chat with automatic provider selection"""
        config = {"purdue": {"api_key": "test-key"}}
        
        with patch('simrag_reproduction.ai_providers.gateway.PurdueGenAI') as mock_purdue, \
             patch('simrag_reproduction.ai_providers.gateway.HuggingFaceClient') as mock_hf, \
             patch('simrag_reproduction.ai_providers.gateway.get_rag_config') as mock_config:
            # Mock config - HuggingFace is default
            mock_config.return_value.use_ollama = False
            mock_config.return_value.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            
            mock_hf_client = MagicMock()
            mock_hf_client.chat.return_value = "Test response"
            mock_hf.return_value = mock_hf_client
            
            gateway = AIGateway(config)
            response = gateway.chat("Hello")
            
            assert response == "Test response"
            # HuggingFace should be used by default
            mock_hf_client.chat.assert_called_once_with("Hello")
    
    def test_chat_with_specific_provider(self):
        """Test chat with specific provider"""
        config = {"purdue": {"api_key": "test-key"}}
        
        with patch('simrag_reproduction.ai_providers.gateway.PurdueGenAI') as mock_purdue:
            mock_client = MagicMock()
            mock_client.chat.return_value = "Test response"
            mock_purdue.return_value = mock_client
            
            gateway = AIGateway(config)
            response = gateway.chat("Hello", provider="purdue", model="test-model")
            
            assert response == "Test response"
            mock_client.chat.assert_called_once_with("Hello", "test-model")
    
    def test_chat_no_providers_available(self):
        """Test chat when no providers are available"""
        with patch('simrag_reproduction.ai_providers.gateway.get_rag_config') as mock_config, \
             patch('simrag_reproduction.ai_providers.gateway.HuggingFaceClient') as mock_hf, \
             patch.dict('os.environ', {}, clear=True):  # Clear environment variables
            # Mock config
            mock_config.return_value.use_ollama = False
            mock_config.return_value.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            
            # Make HuggingFace client fail to initialize
            mock_hf.side_effect = Exception("Failed to load model")
            
            # Gateway should still try to initialize HuggingFace but catch the exception
            # and log a warning (not raise)
            gateway = AIGateway({})
            
            # HuggingFace initialization should have been attempted but failed
            mock_hf.assert_called_once()
            
            # Gateway should have no providers (HuggingFace failed, no others available)
            assert len(gateway.providers) == 0
            
            # Should raise error when trying to chat
            with pytest.raises(Exception, match="No providers available"):
                gateway.chat("Hello")
    
    def test_chat_invalid_provider(self):
        """Test chat with invalid provider"""
        config = {"purdue": {"api_key": "test-key"}}
        
        with patch('simrag_reproduction.ai_providers.gateway.PurdueGenAI'):
            gateway = AIGateway(config)
            
            # Use force_provider=True to ensure exception is raised for invalid provider
            with pytest.raises(Exception, match="Provider 'invalid' not available"):
                gateway.chat("Hello", provider="invalid", force_provider=True)
