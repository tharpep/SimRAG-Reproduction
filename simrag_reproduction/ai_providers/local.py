"""Minimal Ollama client wrapper for local development.

This file provides a small, easy-to-understand Async client for Ollama.
It focuses on a single default model (llama3.2:1b) with clear methods:
- chat(model, messages, **kwargs)
- embeddings(prompt, model=None)
- health_check()
- list_models()

Keep it intentionally small so it's easy to test and extend later.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from .base_client import BaseLLMClient


DEFAULT_MODEL = "llama3.2:1b"


@dataclass
class OllamaConfig:
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    default_model: str = field(default_factory=lambda: os.getenv("MODEL_NAME", DEFAULT_MODEL))
    chat_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_CHAT_TIMEOUT", "15.0")))
    embeddings_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_EMBEDDINGS_TIMEOUT", "30.0")))
    connection_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_CONNECTION_TIMEOUT", "5.0")))


class OllamaClient(BaseLLMClient):
    """Very small Ollama HTTP client.

    Usage:
        async with OllamaClient() as client:
            resp = await client.chat([{"role":"user","content":"Hello"}])
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.logger = logging.getLogger(__name__)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(
                    connect=self.config.connection_timeout,
                    read=max(self.config.chat_timeout, self.config.embeddings_timeout),
                    write=self.config.connection_timeout,
                    pool=self.config.connection_timeout,
                ),
            )
        return self._client

    def chat(self, messages: Any, model: Optional[str] = None, **kwargs) -> str:
        """Send messages to Ollama chat endpoint (sync wrapper for BaseLLMClient).

        Args:
            messages: Chat messages (can be string or list of dicts)
            model: Optional model name; defaults to configured default
        Returns:
            str: AI response text
        """
        import asyncio
        
        # Convert string message to proper format if needed
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        async def _async_chat():
            async with self:
                return await self._async_chat(messages, model, **kwargs)
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        response = loop.run_until_complete(_async_chat())
        return response.get("message", {}).get("content", "")

    async def _async_chat(self, messages: List[Dict[str, Any]], model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Internal async chat method"""
        client = await self._ensure_client()
        model = model or self.config.default_model

        payload = {"model": model, "messages": messages, "stream": False, **kwargs}
        self.logger.debug("ollama chat payload", extra={"model": model, "msg_count": len(messages)})

        resp = await client.post("/api/chat", json=payload, timeout=self.config.chat_timeout)
        resp.raise_for_status()
        return resp.json()

    async def embeddings(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        client = await self._ensure_client()
        model = model or self.config.default_model
        payload = {"model": model, "prompt": prompt}
        resp = await client.post("/api/embeddings", json=payload, timeout=self.config.embeddings_timeout)
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> bool:
        """Check if Ollama server is running and accessible (sync wrapper for BaseLLMClient)"""
        import asyncio
        
        async def _async_health_check():
            try:
                client = await self._ensure_client()
                resp = await client.get("/api/tags", timeout=self.config.connection_timeout)
                return resp.status_code == 200
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return False
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(_async_health_check())

    async def list_models(self) -> List[str]:
        client = await self._ensure_client()
        resp = await client.get("/api/tags", timeout=self.config.connection_timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]
    
    def get_available_models(self) -> List[str]:
        """Sync wrapper for list_models (required by BaseLLMClient)"""
        import asyncio
        async def _get():
            async with self:
                return await self.list_models()
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(_get())