import logging
from typing import List, Dict, Any

from openai import OpenAI
from redis.asyncio import Redis

from .llm_service import LLMService
from ..api.models import ChatRequest, ChatResponse
from ..core.config import settings
from ..knowledge.retriever import KnowledgeRetriever
from ..prompt.manager import PromptManager
from ..utils.helpers import create_message

logger = logging.getLogger(__name__)


def get_knowledge_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()


def get_prompt_manager() -> PromptManager:
    return PromptManager()


class OpenAIService(LLMService):
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.client = OpenAI(api_key=self.api_key)
        self.prompt_manager = get_prompt_manager()
        self.knowledge_retriever = get_knowledge_retriever()

    async def generate_response(self, req: ChatRequest, redis: Redis, **kwargs) -> ChatResponse:
        knowledge = self.knowledge_retriever.get_relevant_documents(req.message, k=3)
        prompt = self.prompt_manager.get_prompt(req.prompt_name)
        history_key = f"chat:history:{req.session_id}"
        history = await redis.lrange(history_key, -5 * 2, -1)
        combined_input = "\n".join(history + [f"Usuario: {req.message}"])

        message = create_message(user_input=combined_input, knowledge=knowledge, prompt=prompt)
        print(message)

        response = self.client.responses.create(
            model=self.model,
            input=message,
            store=False
        )

        output_text = response.output_text

        await redis.rpush(history_key, f"Usuario: {req.message}")
        await redis.rpush(history_key, f"Bot: {output_text}")
        await redis.ltrim(history_key, - (5 * 2), -1)
        await redis.expire(history_key, 3600)

        if not output_text:
            return ChatResponse(response="Lo siento, te conecto con un agente humano.",
                                fallback=True, complete_response=response)
        # if len(output_text) < 100000:
        #    return ChatResponse(response="Lo siento, te conecto con un agente humano.",
        #                        fallback=True, complete_response=response)
        low_quality_phrases = ["no sé", "no puedo", "no estoy seguro"]
        if any(phrase in output_text.lower() for phrase in low_quality_phrases):
            return ChatResponse(response="Lo siento, te conecto con un agente humano.",
                                fallback=True, complete_response=response)

        return ChatResponse(response=output_text, fallback=False, complete_response=response)

    async def list_models(self) -> List[str]:
        return [
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
            "gpt-4-32k", "gpt-3.5-turbo-16k"
        ]

    async def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": "openai"
        }
