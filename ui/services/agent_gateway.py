"""Thin in-process adapter from the Streamlit UI to the assistant."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db.models import Category
from app.db.session import get_session
from app.schemas.chat import ChatTurnResponse
from app.services.chat_runner import graph_values, resume_chat, run_chat

logger = logging.getLogger(__name__)

DEMO_USER_ID = 1
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
_RECEIPT_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf"}


def load_categories(user_id: int = DEMO_USER_ID) -> list[str]:
    """Return the Hebrew category names stored for the demo user."""
    with get_session() as session:
        names = session.scalars(
            select(Category.name).where(Category.user_id == user_id).order_by(Category.name)
        ).all()
    return list(names)


def latest_extraction(thread_id: str) -> dict[str, Any] | None:
    """Return the receipt still held in graph state, if this thread has one."""
    pending = graph_values(thread_id).get("pending_receipt")
    if isinstance(pending, dict):
        return dict(pending)
    return None


def send_turn(
    *,
    thread_id: str,
    message: str,
    upload: tuple[str, bytes] | None = None,
    user_id: int = DEMO_USER_ID,
) -> ChatTurnResponse:
    """Run one chat turn, deleting any temporary receipt file afterwards."""
    path: Path | None = None
    try:
        if upload is not None:
            path = _write_upload(upload[0], upload[1])
        text = message.strip() or "קבלה מצורפת"
        return run_chat(
            user_id=user_id,
            thread_id=thread_id,
            message=text,
            receipt_path=path,
        )
    except Exception:
        logger.exception("Chat turn failed for thread %s", thread_id)
        raise
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def resolve_turn(
    *,
    thread_id: str,
    approved: bool,
    category: str | None = None,
    user_id: int = DEMO_USER_ID,
) -> ChatTurnResponse:
    """Approve, reject, or correct the category of a paused expense."""
    try:
        return resume_chat(
            user_id=user_id,
            thread_id=thread_id,
            approved=approved,
            category=category,
        )
    except Exception:
        logger.exception("Resume failed for thread %s", thread_id)
        raise


def _write_upload(filename: str, payload: bytes) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in _RECEIPT_SUFFIXES:
        raise ValueError("Receipt files must be png, jpg, jpeg, or pdf")
    if not payload:
        raise ValueError("Receipt file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Receipt file is too large")

    descriptor, name = tempfile.mkstemp(suffix=suffix)
    os.close(descriptor)
    path = Path(name)
    try:
        path.write_bytes(payload)
    except Exception:
        path.unlink(missing_ok=True)
        logger.exception("Could not store an uploaded receipt")
        raise
    return path
