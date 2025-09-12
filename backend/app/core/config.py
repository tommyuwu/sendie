from enum import Enum
from typing import Optional

from pydantic.v1 import BaseSettings


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


class Settings(BaseSettings):
    ollama_host: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"

    openai_api_key: Optional[str] = ''
    openai_tommy_key: Optional[str] = ''
    openai_model: str = "gpt-5-nano"

    model_provider: ModelProvider = ModelProvider.OPENAI
    knowledge_base_path: str = "knowledges"
    prompts_dir: str = "prompts"
    default_prompt: str = "default"

    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
