"""
Test Ollama local client
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from ai_providers.local import OllamaClient, OllamaConfig


class TestOllamaConfig:
    """Test cases for OllamaConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = OllamaConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.default_model == "llama3.2:1b"
        assert config.chat_timeout == 15.0
        assert config.embeddings_timeout == 30.0
        assert config.connection_timeout == 5.0
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = OllamaConfig(
            base_url="http://custom:8080",
            default_model="custom-model",
            chat_timeout=30.0
        )
        assert config.base_url == "http://custom:8080"
        assert config.default_model == "custom-model"
        assert config.chat_timeout == 30.0
    
    def test_environment_variable_config(self):
        """Test configuration from environment variables"""
        import os
        with patch.dict(os.environ, {
            'OLLAMA_BASE_URL': 'http://env-test:9999',
            'MODEL_NAME': 'env-model:test',
            'OLLAMA_CHAT_TIMEOUT': '45.0'
        }):
            config = OllamaConfig()
            assert config.base_url == "http://env-test:9999"
            assert config.default_model == "env-model:test"
            assert config.chat_timeout == 45.0


class TestOllamaClient:
    """Test cases for OllamaClient"""
    
    def test_init_default_config(self):
        """Test initialization with default config"""
        client = OllamaClient()
        assert client.config.base_url == "http://localhost:11434"
        assert client.config.default_model == "llama3.2:1b"
        assert client._client is None
    
    def test_init_custom_config(self):
        """Test initialization with custom config"""
        config = OllamaConfig(base_url="http://custom:8080")
        client = OllamaClient(config)
        assert client.config.base_url == "http://custom:8080"
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            async with OllamaClient() as client:
                assert client._client is not None
                mock_client_class.assert_called_once()
            
            # Client should be closed after context
            mock_client.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ensure_client(self):
        """Test client creation"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            client = OllamaClient()
            result = await client._ensure_client()
            
            assert result == mock_client
            assert client._client == mock_client
            mock_client_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_success(self):
        """Test successful chat"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"content": "Test response"}}
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = OllamaClient()
            messages = [{"role": "user", "content": "Hello"}]
            
            response = await client.chat(messages)
            
            assert response == {"message": {"content": "Test response"}}
            mock_client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_custom_model(self):
        """Test chat with custom model"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"content": "Test response"}}
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = OllamaClient()
            messages = [{"role": "user", "content": "Hello"}]
            
            response = await client.chat(messages, model="custom-model")
            
            # Check that custom model was used in the request
            call_args = mock_client.post.call_args
            assert call_args[1]['json']['model'] == "custom-model"
    
    @pytest.mark.asyncio
    async def test_embeddings(self):
        """Test embeddings generation"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = OllamaClient()
            response = await client.embeddings("test prompt")
            
            assert response == {"embedding": [0.1, 0.2, 0.3]}
            mock_client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = OllamaClient()
            result = await client.health_check()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client
            
            client = OllamaClient()
            result = await client.health_check()
            
            assert result is False
    
    
    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing models"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "models": [
                    {"name": "qwen3:1.7b"},
                    {"name": "llama3:latest"}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = OllamaClient()
            models = await client.list_models()
            
            assert models == ["qwen3:1.7b", "llama3:latest"]
    
    def test_get_available_models_sync(self):
        """Test synchronous wrapper for get_available_models"""
        import asyncio
        from unittest.mock import patch, AsyncMock
        
        # Create a real event loop for testing
        loop = asyncio.new_event_loop()
        
        with patch('asyncio.get_event_loop') as mock_get_loop, \
             patch('asyncio.new_event_loop') as mock_new_loop, \
             patch('asyncio.set_event_loop') as mock_set_loop:
            
            mock_get_loop.side_effect = RuntimeError("No event loop")
            mock_new_loop.return_value = loop
            
            # Mock the async context manager and list_models method
            with patch.object(OllamaClient, '__aenter__', new_callable=AsyncMock) as mock_enter, \
                 patch.object(OllamaClient, '__aexit__', new_callable=AsyncMock) as mock_exit, \
                 patch.object(OllamaClient, 'list_models', new_callable=AsyncMock) as mock_list:
                
                mock_list.return_value = ["qwen3:1.7b"]
                mock_enter.return_value = OllamaClient()
                
                client = OllamaClient()
                models = client.get_available_models()
                
                assert models == ["qwen3:1.7b"]
                mock_list.assert_called_once()
        
        loop.close()
