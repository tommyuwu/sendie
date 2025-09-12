import json
import logging

from openai import OpenAI
from redis.asyncio import Redis

from .llm_service import LLMService
from ..api.models import ChatRequest, ChatResponse
from ..core.config import settings
from ..knowledge.retriever import KnowledgeRetriever
from ..prompt.manager import PromptManager
from ..utils.helpers import build_messages

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
        # Prompt y datos relevantes
        knowledge = self.knowledge_retriever.get_relevant_documents(req.message, k=3)
        prompt = self.prompt_manager.get_system_prompt()

        # Contexto de la conversacion
        history_key = f"chat:history:{req.session_id}"
        raw_history = await redis.lrange(history_key, -10, -1)
        history_messages = []
        for line in raw_history:
            if line.startswith("Usuario:"):
                history_messages.append({"role": "user", "content": line[8:].strip()})
            elif line.startswith("Bot:"):
                history_messages.append({"role": "assistant", "content": line[4:].strip()})
        trivial = {"hola", "gracias", "ok", "buenas"}
        if len(req.message.strip()) > 3 and req.message.lower() not in trivial:
            await redis.rpush(history_key, f"Usuario: {req.message}")

        messages = build_messages(
            system_prompt=prompt,
            history=history_messages,
            knowledge=knowledge,
            user_message=req.message
        )

        print(messages)

        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                store=False
            )
        except Exception as e:
            logger.error(f"Error llamando a modelo: {e}")
            return ChatResponse(response="Error interno, derivando a soporte.",
                                fallback=True, complete_response=None)

        output_text = response.output_text

        await redis.rpush(history_key, f"Bot: {output_text}")
        await redis.ltrim(history_key, -10, -1)
        await redis.expire(history_key, 3600)

        if not output_text:
            return ChatResponse(response="Lo siento, te conecto con un agente humano.",
                                fallback=True, complete_response=response)
        low_quality_phrases = ["no sé", "no puedo", "no estoy seguro"]
        if any(phrase in output_text.lower() for phrase in low_quality_phrases):
            return ChatResponse(response="Lo siento, te conecto con un agente humano.",
                                fallback=True, complete_response=response)

        return ChatResponse(response=output_text, fallback=False, complete_response=response)
