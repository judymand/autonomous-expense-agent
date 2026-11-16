"""Create a stable SQLite user and LangGraph thread for each Telegram chat."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Category, ConversationThread, User
from app.db.seed import CATEGORIES

logger = logging.getLogger(__name__)


def ensure_telegram_thread(
    session: Session,
    *,
    chat_id: int,
    telegram_user_id: int,
    display_name: str,
) -> tuple[int, str]:
    """Return ``(user_id, thread_id)``, inserting the chat on first contact.

    ``thread_id`` is stored, not only derived, so a later rename of the formula
    does not split an existing conversation.
    """
    existing = session.scalar(
        select(ConversationThread).where(ConversationThread.telegram_chat_id == chat_id)
    )
    if existing is not None:
        return existing.user_id, existing.thread_id

    user = session.scalar(select(User).where(User.telegram_id == telegram_user_id))
    if user is None:
        user = User(
            telegram_id=telegram_user_id,
            display_name=(display_name.strip() or "Telegram")[:120],
            preferred_currency="ILS",
        )
        session.add(user)
        session.flush()
        session.add_all(
            [Category(user_id=user.id, slug=slug, name=name) for slug, name in CATEGORIES]
        )
        session.flush()
        logger.info("Created Telegram user %s for telegram_id %s", user.id, telegram_user_id)

    thread_id = f"telegram-{chat_id}"
    created = ConversationThread(
        user_id=user.id,
        telegram_chat_id=chat_id,
        thread_id=thread_id,
    )
    session.add(created)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raced = session.scalar(
            select(ConversationThread).where(ConversationThread.telegram_chat_id == chat_id)
        )
        if raced is None:
            raise
        return raced.user_id, raced.thread_id

    logger.info("Mapped Telegram chat %s to thread %s", chat_id, thread_id)
    return user.id, thread_id


def list_categories(session: Session, user_id: int) -> list[tuple[str, str]]:
    """Return ``(slug, Hebrew name)`` pairs in creation order for the keyboard."""
    rows = session.scalars(
        select(Category).where(Category.user_id == user_id).order_by(Category.id)
    ).all()
    return [(row.slug, row.name) for row in rows]
