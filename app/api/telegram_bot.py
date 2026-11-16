"""Telegram polling bot that forwards chats into the LangGraph assistant."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.api.main import load_runtime_env
from app.db.session import get_session, init_db
from app.graph.workflow import build_workflow
from app.services.assistant import AssistantError, AssistantRunner, PendingApproval, SessionFactory
from app.services.telegram_identity import ensure_telegram_thread, list_categories

logger = logging.getLogger(__name__)

START_TEXT = (
    "אפשר לשלוח קבלה (תמונה או PDF), לתאר הוצאה בטקסט, "
    "לשאול על התקציב, או לקבוע כלל קבוע כמו: תמיד לסווג את שופרסל כסופר ומכולת."
)
FAILURE_TEXT = "משהו השתבש. נסו שוב בעוד רגע."
PENDING_TEXT = "יש הוצאה שממתינה לאישור. בחרו כפתור בהודעה הקודמת."


@dataclass(frozen=True)
class BotReply:
    """Text plus the optional approval keyboard to send back to the chat."""

    text: str
    keyboard: InlineKeyboardMarkup | None = None


def build_approval_keyboard(categories: Sequence[tuple[str, str]]) -> InlineKeyboardMarkup:
    """Approve, reject, and one button per category the user can correct to.

    Callback data stays within Telegram's 64-byte limit by using the category
    slug. The button label is the Hebrew name.
    """
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton("Approve ✅", callback_data="approve"),
            InlineKeyboardButton("Reject ❌", callback_data="reject"),
        ]
    ]
    pending: list[InlineKeyboardButton] = []
    for slug, name in categories:
        token = f"cat:{slug}"
        if len(token.encode("utf-8")) > 64:
            logger.warning("Skipping category button %s because the callback is too long", slug)
            continue
        pending.append(InlineKeyboardButton(name, callback_data=token))
        if len(pending) == 2:
            rows.append(pending)
            pending = []
    if pending:
        rows.append(pending)
    return InlineKeyboardMarkup(rows)


def decision_from_callback(
    data: str,
    categories: Sequence[tuple[str, str]],
) -> tuple[bool, str | None] | None:
    """Map a button payload to ``(approved, corrected category name)``."""
    if data == "approve":
        return True, None
    if data == "reject":
        return False, None
    if not data.startswith("cat:"):
        return None
    slug = data.removeprefix("cat:")
    for category_slug, name in categories:
        if category_slug == slug:
            return True, name
    return None


def build_application(
    token: str,
    *,
    runner: AssistantRunner | None = None,
    session_factory: SessionFactory | None = None,
) -> Application:
    """Register text, receipt, and approval-button handlers."""
    application = Application.builder().token(token).build()
    application.bot_data["runner"] = runner if runner is not None else AssistantRunner(build_workflow())
    application.bot_data["session_factory"] = session_factory if session_factory is not None else get_session
    application.add_handler(CommandHandler("start", on_start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(MessageHandler(filters.Document.PDF, on_document))
    application.add_handler(CallbackQueryHandler(on_callback))
    return application


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explain what the bot can do."""
    del context
    if update.message is None:
        return
    await update.message.reply_text(START_TEXT)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route a text message and show approval buttons when the graph pauses."""
    message = update.message
    identity = _identity(update)
    if message is None or identity is None or not message.text:
        return
    chat_id, telegram_user_id, display_name = identity
    try:
        reply = await deliver_turn(
            context,
            chat_id=chat_id,
            telegram_user_id=telegram_user_id,
            display_name=display_name,
            message=message.text,
            receipt_path=None,
        )
    except Exception:
        logger.exception("Telegram text update failed for chat %s", chat_id)
        await message.reply_text(FAILURE_TEXT)
        return
    await message.reply_text(reply.text, reply_markup=reply.keyboard)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download the largest photo and pass it to the graph as a receipt."""
    message = update.message
    identity = _identity(update)
    if message is None or identity is None or not message.photo:
        return
    await _handle_file(
        update,
        context,
        source=message.photo[-1],
        suffix=".jpg",
        caption=message.caption or "קבלה מצורפת",
    )


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download a PDF receipt and pass it to the graph."""
    message = update.message
    identity = _identity(update)
    document = None if message is None else message.document
    if message is None or identity is None or document is None:
        return
    await _handle_file(
        update,
        context,
        source=document,
        suffix=".pdf",
        caption=message.caption or document.file_name or "קבלה מצורפת",
    )


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resume the paused graph from an approval or category button."""
    query = update.callback_query
    identity = _identity(update)
    if query is None or identity is None or not query.data:
        return
    await query.answer()
    chat_id, telegram_user_id, display_name = identity
    try:
        reply = await deliver_callback(
            context,
            chat_id=chat_id,
            telegram_user_id=telegram_user_id,
            display_name=display_name,
            data=query.data,
        )
    except Exception:
        logger.exception("Telegram callback failed for chat %s", chat_id)
        if query.message is not None:
            await query.edit_message_text(FAILURE_TEXT)
        return
    if query.message is not None:
        await query.edit_message_text(reply.text, reply_markup=reply.keyboard)


async def deliver_turn(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    telegram_user_id: int,
    display_name: str,
    message: str,
    receipt_path: Path | None,
) -> BotReply:
    """Resolve the chat in SQLite, run the graph, and format an interrupt keyboard."""
    user_id, thread_id, categories = _bind_chat(
        context,
        chat_id=chat_id,
        telegram_user_id=telegram_user_id,
        display_name=display_name,
    )
    runner = _runner(context)
    try:
        turn = await asyncio.to_thread(
            runner.chat,
            user_id=user_id,
            thread_id=thread_id,
            message=message,
            receipt_path=receipt_path,
        )
    except PendingApproval:
        return BotReply(PENDING_TEXT, build_approval_keyboard(categories))
    except AssistantError:
        logger.exception("Assistant rejected a Telegram turn for chat %s", chat_id)
        return BotReply("לא הצלחתי לטפל בבקשה. נסו לנסח אותה שוב.")
    if turn.status == "interrupted":
        return BotReply(turn.message, build_approval_keyboard(categories))
    return BotReply(turn.message)


async def deliver_callback(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    telegram_user_id: int,
    display_name: str,
    data: str,
) -> BotReply:
    """Turn a button press into ``AssistantRunner.resume``."""
    user_id, thread_id, categories = _bind_chat(
        context,
        chat_id=chat_id,
        telegram_user_id=telegram_user_id,
        display_name=display_name,
    )
    decision = decision_from_callback(data, categories)
    if decision is None:
        return BotReply("הבחירה הזו לא מוכרת. נסו שוב.", build_approval_keyboard(categories))
    approved, category = decision
    try:
        turn = await asyncio.to_thread(
            _runner(context).resume,
            user_id=user_id,
            thread_id=thread_id,
            approved=approved,
            category=category,
        )
    except AssistantError:
        logger.exception("Assistant rejected a Telegram resume for chat %s", chat_id)
        return BotReply("אין הוצאה שממתינה לאישור.")
    if turn.status == "interrupted":
        return BotReply(turn.message, build_approval_keyboard(categories))
    return BotReply(turn.message)


def _bind_chat(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    telegram_user_id: int,
    display_name: str,
) -> tuple[int, str, list[tuple[str, str]]]:
    with _session_factory(context)() as session:
        user_id, thread_id = ensure_telegram_thread(
            session,
            chat_id=chat_id,
            telegram_user_id=telegram_user_id,
            display_name=display_name,
        )
        categories = list_categories(session, user_id)
    return user_id, thread_id, categories


def _runner(context: ContextTypes.DEFAULT_TYPE) -> AssistantRunner:
    runner = context.bot_data.get("runner")
    if not isinstance(runner, AssistantRunner):
        raise RuntimeError("Telegram bot is missing an AssistantRunner")
    return runner


def _session_factory(context: ContextTypes.DEFAULT_TYPE) -> SessionFactory:
    factory = context.bot_data.get("session_factory")
    if factory is None or not callable(factory):
        raise RuntimeError("Telegram bot is missing a session factory")
    return factory


def _identity(update: Update) -> tuple[int, int, str] | None:
    chat = update.effective_chat
    user = update.effective_user
    if chat is None or user is None:
        return None
    display_name = user.full_name or user.username or "Telegram"
    return chat.id, user.id, display_name


async def _handle_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    source: object,
    suffix: str,
    caption: str,
) -> None:
    message = update.message
    identity = _identity(update)
    if message is None or identity is None:
        return
    chat_id, telegram_user_id, display_name = identity
    path: Path | None = None
    try:
        path = await _download(source, suffix)
        reply = await deliver_turn(
            context,
            chat_id=chat_id,
            telegram_user_id=telegram_user_id,
            display_name=display_name,
            message=caption,
            receipt_path=path,
        )
    except Exception:
        logger.exception("Telegram file update failed for chat %s", chat_id)
        await message.reply_text(FAILURE_TEXT)
        return
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
    await message.reply_text(reply.text, reply_markup=reply.keyboard)


async def _download(source: object, suffix: str) -> Path:
    get_file = getattr(source, "get_file")
    telegram_file = await get_file()
    descriptor, name = tempfile.mkstemp(suffix=suffix)
    os.close(descriptor)
    path = Path(name)
    try:
        await telegram_file.download_to_drive(custom_path=str(path))
    except Exception:
        path.unlink(missing_ok=True)
        logger.exception("Failed to download a Telegram receipt")
        raise
    return path


def main() -> None:
    """Load the environment, prepare SQLite, and poll Telegram."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_runtime_env()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required to start the Telegram bot")
    try:
        init_db()
        application = build_application(token)
    except Exception:
        logger.exception("Failed to start the Telegram bot")
        raise
    logger.info("Telegram bot polling")
    application.run_polling()


if __name__ == "__main__":
    main()
