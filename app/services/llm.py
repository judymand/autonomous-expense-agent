"""Model-agnostic chat-model factory.

``LLM_PROVIDER=google`` builds Gemini through ``langchain-google-genai``.
``LLM_PROVIDER=ollama`` builds a local Ollama chat model. Model names and the
API key come from the environment or ``.env``.
"""

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.db.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

ProviderName = Literal["google", "ollama"]
_SECRET_KEYS = (
    "GOOGLE_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "DATABASE_URL",
    "CHROMA_PERSIST_DIR",
)
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
# Tried in order when the primary Gemini model times out or returns an error.
# Names that match ``LLM_MODEL`` are skipped so the chain never repeats a model.
GEMINI_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
)
# Catch provider outages such as ``503 UNAVAILABLE`` and ``GoogleAPIError``.
_FALLBACK_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)
GEMINI_TIMEOUT_SECONDS = 60
GEMINI_MAX_RETRIES = 1
DEFAULT_OLLAMA_MODEL = "llava"
ChatModel = BaseChatModel | Runnable[Any, Any]


class _GeminiWithFallbacks(RunnableWithFallbacks[Any, Any]):
    """Gemini chain that reapplies structured output to every fallback model.

    ``RunnableWithFallbacks`` only keeps the fallback list for methods it can
    prove return a ``Runnable``. Nodes such as the advisor call
    ``with_structured_output`` and must still try the next model when Gemini
    raises ``503 UNAVAILABLE`` or ``GoogleAPIError``.
    """

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        """Bind ``schema`` on the primary model and on each fallback."""
        primary = self.runnable.with_structured_output(schema, **kwargs)
        fallbacks = [
            fallback.with_structured_output(schema, **kwargs) for fallback in self.fallbacks
        ]
        if not fallbacks:
            return primary
        return primary.with_fallbacks(fallbacks, exceptions_to_handle=_FALLBACK_EXCEPTIONS)


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


def prime_llm_environment() -> None:
    """Load ``.env`` and Streamlit secrets into the process environment.

    Local ``.env`` values win. Streamlit Community Cloud secrets fill anything
    still missing, either from ``.streamlit/secrets.toml`` or from ``st.secrets``
    when the Streamlit runtime is already imported. Cached settings are dropped
    so the next model build sees the merged values.
    """
    from dotenv import load_dotenv

    from app.db.settings import get_settings

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    _load_toml_secrets(PROJECT_ROOT / ".streamlit" / "secrets.toml")
    _load_streamlit_runtime_secrets()
    get_llm_settings.cache_clear()
    get_settings.cache_clear()


def _load_toml_secrets(path: Path) -> None:
    if not path.is_file():
        return
    try:
        import tomllib

        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Could not read Streamlit secrets at %s", path)
        return
    if isinstance(payload, dict):
        _export_mapping(payload)


def _load_streamlit_runtime_secrets() -> None:
    streamlit = sys.modules.get("streamlit")
    if streamlit is None:
        return
    try:
        secrets = getattr(streamlit, "secrets")
    except Exception:
        logger.info("Streamlit secrets are unavailable")
        return
    try:
        mapping = dict(secrets)
    except Exception:
        logger.debug("Streamlit secrets are not configured")
        return
    _export_mapping(mapping)


def _export_mapping(mapping: dict[object, object]) -> None:
    for key in _SECRET_KEYS:
        if os.getenv(key):
            continue
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            os.environ[key] = text


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """Return the process-wide LLM settings."""
    return LLMSettings()


def get_chat_model(settings: LLMSettings | None = None) -> ChatModel:
    """Build the configured chat model without calling it.

    When ``LLM_MODEL`` is unset, Gemini uses ``gemini-3.5-flash-lite``, then
    ``gemini-3.1-flash-lite``, ``gemini-3.5-flash``, and ``gemini-3.6-flash`` if
    that call fails. A configured ``LLM_MODEL`` stays first and is removed from
    the fallback list.
    """
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


def _google_model(cfg: LLMSettings, model_name: str) -> ChatModel:
    if cfg.google_api_key is None:
        raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=google")

    primary = _gemini_chat(cfg, model_name)
    fallback_names = [name for name in GEMINI_FALLBACK_MODELS if name != model_name]
    fallbacks = [_gemini_chat(cfg, name) for name in fallback_names]
    logger.info("Creating Google chat model %s with fallbacks %s", model_name, fallback_names)
    if not fallbacks:
        return primary
    chain = primary.with_fallbacks(fallbacks, exceptions_to_handle=_FALLBACK_EXCEPTIONS)
    return _GeminiWithFallbacks(
        runnable=chain.runnable,
        fallbacks=list(chain.fallbacks),
        exceptions_to_handle=_FALLBACK_EXCEPTIONS,
    )


def _gemini_chat(cfg: LLMSettings, model_name: str) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if cfg.google_api_key is None:
        raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=google")

    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=cfg.google_api_key.get_secret_value(),
        temperature=0,
        max_retries=GEMINI_MAX_RETRIES,
        timeout=GEMINI_TIMEOUT_SECONDS,
    )


def _ollama_model(cfg: LLMSettings, model_name: str) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    logger.info("Creating Ollama chat model %s at %s", model_name, cfg.ollama_base_url)
    return ChatOllama(
        model=model_name,
        base_url=cfg.ollama_base_url,
        temperature=0,
    )
