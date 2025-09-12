import logging
from typing import Dict

from fastapi import APIRouter, HTTPException, Depends

from .models import HealthResponse
from ..core.config import ModelProvider, settings
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
        llm_service: LLMService = Depends(get_llm_service),
        knowledge_retriever: KnowledgeRetriever = Depends(get_knowledge_retriever),
        prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    try:
        models = await llm_service.list_models()
        model_info = await llm_service.get_model_info()

        return HealthResponse(
            status="healthy",
            model_provider="",
            model=model_info.get("model", "unknown"),
            models_available=models,
            knowledge_base_loaded=knowledge_retriever.vectorstore is not None,
            available_prompts=list(prompt_manager.prompts.keys())
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@router.get("/prompts", response_model=Dict[str, str])
async def list_prompts(
        prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    return prompt_manager.list_prompts()


@router.get("/prompts/{prompt_name}", response_model=Dict[str, str])
async def get_prompt(
        prompt_name: str,
        prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    prompt = prompt_manager.get_prompt(prompt_name)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")
    return {"name": prompt_name, "content": prompt}


@router.post("/chat/switch-provider")
async def switch_provider(
        provider: ModelProvider
):
    settings.model_provider = provider
    return {"message": f"Provider switched to {provider.value}"}
