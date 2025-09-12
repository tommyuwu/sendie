from abc import ABC, abstractmethod
from typing import List, Dict, Any

from redis.asyncio import Redis

from ..api.models import ChatRequest


class LLMService(ABC):

    @abstractmethod
    async def generate_response(self, req: ChatRequest, redis: Redis, **kwargs) -> str:
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        pass

    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        pass
