import logging

from fastapi import HTTPException, Depends, APIRouter, Request
from redis.asyncio import Redis

from ..api.models import ChatResponse, ChatRequest
from ..service.factory import LLMServiceFactory
from ..service.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_llm_service() -> LLMService:
    return LLMServiceFactory.create_service()


async def get_redis_client(request: Request) -> Redis:
    return request.app.state.redis


@router.post("/chat", response_model=ChatResponse)
async def chat(
        req: ChatRequest,
        llms: LLMService = Depends(get_llm_service),
        redis: Redis = Depends(get_redis_client)
):
    try:
        return await llms.generate_response(req=req, redis=redis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al llamar Responses API: {str(e)}")
