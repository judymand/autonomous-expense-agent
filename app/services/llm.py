"""Model-agnostic chat-model factory.

``LLM_PROVIDER=google`` builds Gemini through ``langchain-google-genai``.
``LLM_PROVIDER=ollama`` builds a local Ollama chat model. Model names and the
API key come from the environment or ``.env``.
"""

import logging
from functools import lru_cache
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.db.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

ProviderName = Literal["google", "ollama"]
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OLLAMA_MODEL = "llava"


class LLMSettings(BaseSettings):
    """Provider selection for the shared chat model."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: ProviderName = "google"
    llm_model: str = Field(default=DEFAULT_GEMINI_MODEL)
    google_api_key: SecretStr | None = None
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = DEFAULT_OLLAMA_MODEL

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("google_api_key", mode="before")
    @classmethod
    def _blank_api_key(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def resolved_model(self) -> str:
        """Return the model name for the active provider.

        Gemini uses ``LLM_MODEL``. Ollama uses ``OLLAMA_MODEL`` so a Gemini model
        name left in ``LLM_MODEL`` is not sent to the local server.
        """
        if self.llm_provider == "ollama":
            return self.ollama_model
        return self.llm_model


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """Return the process-wide LLM settings."""
    return LLMSettings()


def get_chat_model(settings: LLMSettings | None = None) -> BaseChatModel:
    """Build the configured chat model without calling it."""
    cfg = settings if settings is not None else get_llm_settings()
    model_name = cfg.resolved_model()
    try:
        if cfg.llm_provider == "google":
            return _google_model(cfg, model_name)
        if cfg.llm_provider == "ollama":
            return _ollama_model(cfg, model_name)
    except Exception:
        logger.exception("Failed to create the %s chat model", cfg.llm_provider)
        raise
    raise ValueError(f"Unsupported LLM provider: {cfg.llm_provider}")


def _google_model(cfg: LLMSettings, model_name: str) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if cfg.google_api_key is None:
        raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=google")

    logger.info("Creating Google chat model %s", model_name)
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=cfg.google_api_key.get_secret_value(),
        temperature=0,
    )


def _ollama_model(cfg: LLMSettings, model_name: str) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    logger.info("Creating Ollama chat model %s at %s", model_name, cfg.ollama_base_url)
    return ChatOllama(
        model=model_name,
        base_url=cfg.ollama_base_url,
        temperature=0,
    )
