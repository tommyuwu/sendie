from abc import ABC, abstractmethod

from redis.asyncio import Redis

from ..api.models import ChatRequest


class LLMService(ABC):

    @abstractmethod
    async def generate_response(self, req: ChatRequest, redis: Redis, **kwargs) -> str:
        pass
