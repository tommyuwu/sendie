import logging

from fastapi import APIRouter, HTTPException, Depends

from .models import HealthResponse
from ..knowledge.retriever import KnowledgeRetriever
from ..prompt.manager import PromptManager
from ..service.factory import LLMServiceFactory
from ..service.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_llm_service() -> LLMService:
    return LLMServiceFactory.create_service()


def get_knowledge_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()


def get_prompt_manager() -> PromptManager:
    return PromptManager()


@router.get("/health", response_model=HealthResponse)
async def health_check(
        knowledge_retriever: KnowledgeRetriever = Depends(get_knowledge_retriever),
        prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    try:

        return HealthResponse(
            status="healthy",
            model_provider="",
            knowledge_base_loaded=knowledge_retriever.vectorstore is not None,
            available_prompts=list(prompt_manager.prompts.keys())
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
