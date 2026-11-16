"""Dependencies the graph nodes read from the runnable config."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.db.vector_store import get_user_preferences_collection
from app.services.llm import get_chat_model

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


def configurable(config: RunnableConfig | None) -> dict[str, object]:
    """Return the configurable mapping, or an empty dict."""
    if not config:
        return {}
    raw = config.get("configurable", {})
    if isinstance(raw, dict):
        return raw
    return {}


def llm_from_config(config: RunnableConfig | None) -> object:
    """Use the model injected by tests, otherwise the shared chat model."""
    override = configurable(config).get("llm")
    if override is not None:
        return override
    return get_chat_model()


@contextmanager
def session_scope(config: RunnableConfig | None) -> Iterator[Session]:
    """Yield the injected session, or open and close the application session."""
    existing = configurable(config).get("session")
    if isinstance(existing, Session):
        yield existing
        return

    with get_session() as session:
        yield session


def preferences_from_config(config: RunnableConfig | None) -> object:
    """Use the injected Chroma collection, or the persistent user_preferences collection."""
    override = configurable(config).get("preferences")
    if override is not None:
        return override
    try:
        return get_user_preferences_collection()
    except Exception:
        logger.exception("Could not open the user_preferences collection")
        raise


def as_model(parsed: object, schema: type[ModelT]) -> ModelT:
    """Accept either a model instance or a mapping from structured output."""
    if isinstance(parsed, schema):
        return parsed
    return schema.model_validate(parsed)
