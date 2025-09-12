import logging

import ollama

from .llm_service import LLMService
from ..api.models import ChatRequest
from ..core.config import settings
from ..knowledge.retriever import KnowledgeRetriever
from ..prompt.manager import PromptManager

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
            prompt = self.prompt_manager.get_system_prompt()
            # message = build_messages(user_input=req.message, knowledge=knowledge, prompt=prompt)
            message = ''
            response = self.client.generate(
                model=self.model,
                prompt=message,
                options=kwargs
            )
            return response["response"]
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            raise
