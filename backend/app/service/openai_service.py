import logging
import json

from openai import OpenAI
from openai.types.responses import Response
from redis.asyncio import Redis

from .llm_service import LLMService
from ..api.models import ChatRequest, ChatResponse
from ..core.config import settings
from ..knowledge.retriever import KnowledgeRetriever
from ..message_manager.manager import MessageManager
from ..prompt.manager import PromptManager
from ..utils.fallback_pipeline import build_fallback

logger = logging.getLogger(__name__)


def get_knowledge_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()


def get_prompt_manager() -> PromptManager:
    return PromptManager()


def get_message_manager() -> MessageManager:
    return MessageManager()


def needs_fallback(output_text: str, response: Response) -> bool:
    if not output_text or len(output_text.strip()) < 5:
        return True
    if "no tengo" in output_text.lower() or "desconozco" in output_text.lower():
        return True

    return False


def fallback_response(response):
    return ChatResponse(
        response="Lo siento, no pude procesar tu consulta. Te conecto con un agente humano.",
        fallback=True,
        complete_response=response
    )


class OpenAIService(LLMService):
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.client = OpenAI(api_key=self.api_key)
        self.knowledge_retriever = get_knowledge_retriever()
        self.prompt_manager = get_prompt_manager()
        self.message_manager = get_message_manager()

    async def generate_response(self, req: ChatRequest, redis: Redis, **kwargs) -> ChatResponse:
        try:

            # knowledge = self.knowledge_retriever.get_relevant_documents(req.message, k=5)
            # prompt = self.prompt_manager.get_system_prompt()
            # TODO: falta mejorar el manejo de los ultimos mensajes
            history = await self.message_manager.get_history(req.session_id, redis)

            # message = _enrich_query_with_context(req.message, knowledge)

            # final_message = build_messages(
            #    system_prompt=prompt,
            #    knowledge=knowledge,
            #    history=history,
            #    user_message=message
            # )

            # return None
            response = self.client.responses.create(
                input=req.message,
                store=False,
                prompt={
                    "id": settings.openai_pmpt_id
                }
            )
            parsed = json.loads(response.output_text)
            output_text = parsed.get("answer", "").strip()
            fallback_signal = parsed.get("fallback_signal", "LOW_CONFIDENCE")

            if fallback_signal == "NONE":
                await self.message_manager.push_to_history(req, output_text, redis)
                return ChatResponse(
                    response=output_text,
                    fallback=False,
                    complete_response=response
                )
            else:
                return build_fallback(fallback_signal, response)

        except Exception as e:
            logger.error(f"Error generando respuesta: {e}", exc_info=True)
            return ChatResponse(
                response="Disculpa, tuve un problema técnico. Te conecto con un agente humano.",
                fallback=True,
                complete_response=None
            )
