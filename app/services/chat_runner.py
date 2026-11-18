"""In-process chat entry points shared by the API and the Streamlit UI.

The compiled graph and its SQLite checkpointer live for the life of the
process. Streamlit Community Cloud can call these functions directly, without
a separate FastAPI server.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from app.db.session import init_db
from app.graph.workflow import build_workflow
from app.schemas.chat import ChatTurnResponse
from app.services.assistant import AssistantRunner
from app.services.llm import prime_llm_environment

logger = logging.getLogger(__name__)

_runner: AssistantRunner | None = None
_lock = threading.Lock()


def run_chat(
    *,
    user_id: int,
    thread_id: str,
    message: str,
    receipt_path: str | Path | None = None,
) -> ChatTurnResponse:
    """Run one user turn and include the optional node trace."""
    return get_runner().chat(
        user_id=user_id,
        thread_id=thread_id,
        message=message,
        receipt_path=receipt_path,
    )


def resume_chat(
    *,
    user_id: int,
    thread_id: str,
    approved: bool,
    category: str | None = None,
) -> ChatTurnResponse:
    """Resume a paused expense and include the optional node trace."""
    return get_runner().resume(
        user_id=user_id,
        thread_id=thread_id,
        approved=approved,
        category=category,
    )


def graph_values(thread_id: str) -> dict[str, Any]:
    """Return the latest checkpoint values for a thread."""
    snapshot = get_runner().graph.get_state({"configurable": {"thread_id": thread_id}})
    raw = getattr(snapshot, "values", None)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def get_runner() -> AssistantRunner:
    """Build the shared assistant once, after secrets and the schema exist."""
    global _runner
    with _lock:
        if _runner is None:
            prime_llm_environment()
            init_db()
            _runner = AssistantRunner(build_workflow())
            logger.info("In-process assistant runner is ready")
        return _runner
