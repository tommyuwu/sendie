from typing import Optional

from .llm_service import LLMService
from .ollama_service import OllamaService
from .openai_service import OpenAIService
from ..core.config import settings, ModelProvider


class LLMServiceFactory:
    @staticmethod
    def create_service(
            provider: Optional[ModelProvider] = None
    ) -> LLMService:
        provider = provider or settings.model_provider

        if provider == ModelProvider.OLLAMA:
            return OllamaService()
        elif provider == ModelProvider.OPENAI:
            return OpenAIService()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
