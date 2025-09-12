import logging
from typing import List, Dict, Any

import ollama

from .llm_service import LLMService
from ..api.models import ChatRequest
from ..core.config import settings
from ..knowledge.retriever import KnowledgeRetriever
from ..prompt.manager import PromptManager
from ..utils.helpers import create_message

logger = logging.getLogger(__name__)


def get_knowledge_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()


def get_prompt_manager() -> PromptManager:
    return PromptManager()


class OllamaService(LLMService):
    def __init__(self):
        self.host = settings.ollama_host
        self.model = settings.ollama_model
        self.client = ollama.Client(host=self.host)
        self.prompt_manager = get_prompt_manager()
        self.knowledge_retriever = get_knowledge_retriever()

    async def generate_response(self, req: ChatRequest, **kwargs) -> str:
        try:
            knowledge = self.knowledge_retriever.get_relevant_documents(req.message, k=3)
            prompt = self.prompt_manager.get_prompt(req.prompt_name)
            message = create_message(user_input=req.message, knowledge=knowledge, prompt=prompt)

            response = self.client.generate(
                model=self.model,
                prompt=message,
                options=kwargs
            )
            return response["response"]
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            raise

    async def list_models(self) -> List[str]:
        try:
            models = self.client.list()
            return [model["model"] for model in models.get("models", [])]
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    async def get_model_info(self) -> Dict[str, Any]:
        try:
            models = self.client.list()
            for model in models.get("models", []):
                if model["model"] == self.model:
                    return model
            return {}
        except Exception as e:
            logger.error(f"Error getting Ollama model info: {e}")
            return {}
