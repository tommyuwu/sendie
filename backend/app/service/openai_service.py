import logging
import re
from typing import List

from openai import OpenAI
from redis.asyncio import Redis

from .llm_service import LLMService
from ..api.models import ChatRequest, ChatResponse
from ..core.config import settings
from ..knowledge.retriever import KnowledgeRetriever
from ..message_manager.manager import MessageManager
from ..prompt.manager import PromptManager
from ..utils.helpers import build_messages

logger = logging.getLogger(__name__)


def get_knowledge_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()


def get_prompt_manager() -> PromptManager:
    return PromptManager()


def get_message_manager() -> MessageManager:
    return MessageManager()


def _calculate_shipping_cost(weight: float, location: str, shipping_type: str, knowledge: List[str]) -> str | None:
    for doc in knowledge:
        if 'TARIFAS' in doc and shipping_type.upper() in doc:
            lines = doc.split('\n')
            for line in lines:
                if location and location.lower() in line.lower():
                    match = re.search(r'\$(\d+(?:\.\d+)?)', line)
                    if match:
                        price_per_kg = float(match.group(1))
                        total_cost = weight * price_per_kg

                        if shipping_type == 'aéreo' and weight < 0.13:
                            return f"Para envío aéreo a {location}, el peso mínimo es 130g (0.13kg). El costo mínimo sería ${price_per_kg * 0.13:.2f} USD."

                        return f"El costo de envío {shipping_type} de {weight}kg a {location} sería: ${total_cost:.2f} USD (tarifa: ${price_per_kg} USD/kg)."

    return None


def _extract_weight_from_query(query: str) -> float | None:
    patterns = [
        r'(\d+(?:\.\d+)?)\s*kg',
        r'(\d+(?:\.\d+)?)\s*kilos?',
        r'(\d+(?:\.\d+)?)\s*gramos?',
        r'(\d+(?:\.\d+)?)\s*g\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            weight = float(match.group(1))
            if 'g' in pattern and 'kg' not in pattern:
                weight = weight / 1000
            return weight

    return None


def _extract_location_from_query(query: str) -> str | None:
    query_lower = query.lower()

    sucursales = [
        'alberdi', 'asunción', 'asuncion', 'chaco filadelfia', 'ciudad del este',
        'coronel oviedo', 'encarnacion', 'encarnación', 'lambare', 'lambaré',
        'loma plata', 'luque', 'm4', 'mariano roque alonso', 'nanawa',
        'pedro juan caballero', 'salto del guaira', 'salto del guairá',
        'san lorenzo', 'villarrica'
    ]

    for sucursal in sucursales:
        if sucursal in query_lower:
            if sucursal in ['asuncion', 'asunción']:
                return 'Asunción'
            elif sucursal in ['lambare', 'lambaré']:
                return 'Lambaré'
            elif sucursal in ['encarnacion', 'encarnación']:
                return 'Encarnación'
            else:
                return ' '.join(word.capitalize() for word in sucursal.split())

    return None


def _enrich_query_with_context(query: str, knowledge: List[str]) -> str:
    query_lower = query.lower()

    context_additions = []

    if any(word in query_lower for word in ['tarifa', 'precio', 'costo', 'cuánto', 'cuanto', 'sale']):
        weight = _extract_weight_from_query(query)
        location = _extract_location_from_query(query)

        shipping_type = 'aéreo' if 'aéreo' in query_lower or 'aereo' in query_lower else None
        shipping_type = 'marítimo' if 'marítimo' in query_lower or 'maritimo' in query_lower else shipping_type

        if weight and location and shipping_type:
            cost_info = _calculate_shipping_cost(weight, location, shipping_type, knowledge)
            if cost_info:
                context_additions.append(f"Información calculada: {cost_info}")

    if any(word in query_lower for word in ['tiempo', 'demora', 'días', 'cuando', 'cuándo']):
        for doc in knowledge:
            if 'tiempo estimado' in doc.lower():
                context_additions.append(f"Información de tiempos: {doc}")
                break

    if context_additions:
        return query + "\n\nContexto adicional:\n" + "\n".join(context_additions)

    return query


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

            knowledge = self.knowledge_retriever.get_relevant_documents(req.message, k=5)
            prompt = self.prompt_manager.get_system_prompt()
            history = await self.message_manager.get_history(req.session_id, redis)

            message = _enrich_query_with_context(req.message, knowledge)

            final_message = build_messages(
                system_prompt=prompt,
                knowledge=knowledge,
                history=history,
                user_message=message
            )
            logger.info(f"Mensaje final: {final_message}")

            #return None
            response = self.client.responses.create(
                model=self.model,
                input=final_message,
                store=False,
                # reasoning={"effort": "low"},
                max_output_tokens=2000
            )

            output_text = response.output_text

            if output_text:
                await self.message_manager.push_to_history(req, output_text, redis)
                return ChatResponse(
                    response=output_text,
                    fallback=False,
                    complete_response=response
                )
            else:
                return ChatResponse(
                    response="Lo siento, no pude procesar tu consulta. Te conecto con un agente humano.",
                    fallback=True,
                    complete_response=response
                )

        except Exception as e:
            logger.error(f"Error generando respuesta: {e}", exc_info=True)
            return ChatResponse(
                response="Disculpa, tuve un problema técnico. Te conecto con un agente humano.",
                fallback=True,
                complete_response=None
            )
