from enum import Enum
from typing import Optional

from pydantic.v1 import BaseSettings


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


class Settings(BaseSettings):
    ollama_host: str = ''
    ollama_model: str = ''

    openai_api_key: Optional[str] = ''
    openai_model: str = ''

    title: str = ''
    model_provider: ModelProvider = ModelProvider.OPENAI
    knowledge_base_path: str = ''
    knowledge_base_json_file: str = ''
    prompts_dir: str = ''
    system_prompt: str = ''

    redis_host: str = ''
    redis_port: int = 0
    redis_db: int = 0

    max_history_turns: int = 5
    max_prompt_tokens: int = 3000
    max_doc_tokens: int = 500
    max_doc_chars = 600

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
