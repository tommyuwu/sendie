import logging
from pathlib import Path
from typing import Dict

from ..core.config import settings

logger = logging.getLogger(__name__)


class PromptManager:
    def __init__(self):
        self.prompts_dir = settings.prompts_dir
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompts = {}
        prompts_path = Path(self.prompts_dir)
        if not prompts_path.exists():
            logger.warning(f"Prompts directory {self.prompts_dir} does not exist")
            return prompts

        for prompt_file in prompts_path.glob("*.txt"):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompts[prompt_file.stem] = f.read().strip()
                logger.info(f"Loaded prompt: {prompt_file.stem}")
            except Exception as e:
                logger.error(f"Error loading prompt {prompt_file}: {e}")

        return prompts

    def get_system_prompt(self) -> str:
        return self.prompts.get(settings.system_prompt)
