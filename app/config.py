from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    SERVICE_NAME: str = "proofsanalyze"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    ALLOW_ORIGINS: str = "*"

    LLM_BACKEND: str = "ollama"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4000

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_TIMEOUT: float = 180.0

    JOB_STORE_TTL_SECONDS: int = Field(default=3600, ge=60)

    @property
    def allow_origins_list(self) -> list[str]:
        if self.ALLOW_ORIGINS.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.ALLOW_ORIGINS.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR
