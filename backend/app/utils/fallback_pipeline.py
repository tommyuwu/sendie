from openai.types.responses import Response
from ..api.models import ChatResponse


def build_fallback(reason: str, response: Response) -> ChatResponse:
    if reason == "REQUEST_HUMAN":
        return ChatResponse(
            response="Te estamos contactando con un agente humano, aguarde un momento.",
            fallback=True,
            complete_response=response
        )
    elif reason == "LOW_CONFIDENCE":
        return ChatResponse(
            response="Te estamos contactando con un agente humano, aguarde un momento.",
            fallback=True,
            complete_response=response
        )
    else:
        return ChatResponse(
            response="Te estamos contactando con un agente humano, aguarde un momento.",
            fallback=True,
            complete_response=response
        )
