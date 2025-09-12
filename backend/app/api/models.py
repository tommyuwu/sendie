from typing import List, Optional, TypeVar, Generic

from pydantic import BaseModel

T = TypeVar('T')


class ChatRequest(BaseModel):
    session_id: str
    message: str
    prompt_name: Optional[str] = 'default'


class ChatResponse(BaseModel, Generic[T]):
    response: str
    fallback: bool = False
    complete_response: T


class HealthResponse(BaseModel):
    status: str
    model_provider: str
    model: str
    models_available: List[str]
    knowledge_base_loaded: bool
    available_prompts: List[str]
