"""Connectivity checks for the API health endpoint."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

from sqlalchemy import Engine, text

from app.db.session import get_engine
from app.db.vector_store import get_chroma_client

logger = logging.getLogger(__name__)


def probe_sqlite(engine: Engine | None = None) -> None:
    """Run a trivial query so a missing or locked database fails the check."""
    target = engine if engine is not None else get_engine()
    with target.connect() as connection:
        connection.execute(text("SELECT 1"))


def probe_chroma(client: object | None = None) -> None:
    """Ping ChromaDB. Creating a collection is intentionally not part of the check."""
    chroma = client if client is not None else get_chroma_client()
    heartbeat: Callable[[], object] = getattr(chroma, "heartbeat")
    heartbeat()


def probe_status(check: Callable[[], None], name: str) -> Literal["ok", "error"]:
    """Return ``ok`` or ``error`` without letting a dependency outage escape."""
    try:
        check()
    except Exception:
        logger.exception("%s health check failed", name)
        return "error"
    return "ok"
