"""Run one LangGraph turn and read an approval interrupt back out of the checkpoint."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import get_session
from app.schemas.chat import ChatTurnResponse, TurnTrace
from app.services.turn_trace import build_turn_trace, checkpoint_id

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractContextManager[Session]]

_REASON_TEXT = {
    "amount_over_500": "הסכום גבוה מ-500 ש״ח",
    "low_confidence": "הביטחון בזיהוי נמוך",
    "duplicate": "זו נראית כמו הוצאה שכבר נשמרה",
}


class AssistantError(Exception):
    """A turn the API can turn into an HTTP status without a traceback."""

    def __init__(self, detail: str, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class UnknownUser(AssistantError):
    """The caller named a user that is not in SQLite."""

    def __init__(self, user_id: int) -> None:
        super().__init__(f"User {user_id} was not found", 404)


class PendingApproval(AssistantError):
    """A new message arrived while this thread is still waiting for a decision."""

    def __init__(self, thread_id: str) -> None:
        super().__init__(
            f"Thread {thread_id} is waiting for approval. Resume it before sending another message.",
            409,
        )


class NothingToResume(AssistantError):
    """Resume was called when the thread has no pending interrupt."""

    def __init__(self, thread_id: str) -> None:
        super().__init__(f"Thread {thread_id} has no expense waiting for approval", 409)


class ThreadUserMismatch(AssistantError):
    """The resume caller does not own the paused thread."""

    def __init__(self) -> None:
        super().__init__("This thread belongs to a different user", 403)


class AssistantRunner:
    """Invoke the compiled graph with a checkpointer thread id.

    A lock serializes turns because the SQLite checkpointer connection is shared.
    The session is opened inside the call so it stays on the worker thread.
    """

    def __init__(
        self,
        graph: Any,
        *,
        session_factory: SessionFactory | None = None,
        llm: object | None = None,
        preferences: object | None = None,
    ) -> None:
        self.graph = graph
        self.session_factory = session_factory if session_factory is not None else get_session
        self.llm = llm
        self.preferences = preferences
        self._lock = threading.Lock()

    def chat(
        self,
        *,
        user_id: int,
        thread_id: str,
        message: str,
        receipt_path: str | Path | None = None,
    ) -> ChatTurnResponse:
        """Append a user message and return the reply or the paused receipt."""
        with self._lock:
            if _interrupt_payload(self._snapshot(thread_id)) is not None:
                raise PendingApproval(thread_id)

            turn: ChatTurnResponse | None = None
            with self.session_factory() as session:
                if session.get(User, user_id) is None:
                    found = False
                else:
                    found = True
                    before = checkpoint_id(self._snapshot(thread_id))
                    result = self.graph.invoke(
                        _graph_input(user_id, message, receipt_path),
                        self._config(thread_id, session),
                    )
                    snapshot = self._snapshot(thread_id)
                    turn = _turn_from_snapshot(
                        user_id,
                        thread_id,
                        snapshot,
                        result,
                        trace=build_turn_trace(self.graph, thread_id, before, _values(snapshot)),
                    )
            if not found or turn is None:
                raise UnknownUser(user_id)
            logger.info("Thread %s finished with status %s", thread_id, turn.status)
            return turn

    def resume(
        self,
        *,
        user_id: int,
        thread_id: str,
        approved: bool,
        category: str | None = None,
    ) -> ChatTurnResponse:
        """Continue a paused expense with an approval decision."""
        with self._lock:
            snapshot = self._snapshot(thread_id)
            if _interrupt_payload(snapshot) is None:
                raise NothingToResume(thread_id)
            owner = _values(snapshot).get("user_id")
            if owner != user_id:
                raise ThreadUserMismatch()

            decision: dict[str, object] = {"approved": approved}
            if category:
                decision["category"] = category

            before = checkpoint_id(snapshot)
            with self.session_factory() as session:
                result = self.graph.invoke(
                    Command(resume=decision),
                    self._config(thread_id, session),
                )
                resumed = self._snapshot(thread_id)
                turn = _turn_from_snapshot(
                    user_id,
                    thread_id,
                    resumed,
                    result,
                    trace=build_turn_trace(self.graph, thread_id, before, _values(resumed)),
                )
            logger.info("Thread %s resumed with status %s", thread_id, turn.status)
            return turn

    def _snapshot(self, thread_id: str) -> object:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        return self.graph.get_state(config)

    def _config(self, thread_id: str, session: Session) -> RunnableConfig:
        configurable: dict[str, object] = {"thread_id": thread_id, "session": session}
        if self.llm is not None:
            configurable["llm"] = self.llm
        if self.preferences is not None:
            configurable["preferences"] = self.preferences
        return {"configurable": configurable}


def format_interrupt_message(payload: Mapping[str, Any]) -> str:
    """Hebrew text that explains why an expense is waiting and how to decide."""
    merchant = str(payload.get("merchant_name") or "העסק")
    amount = str(payload.get("total_amount") or "").strip()
    currency = str(payload.get("currency") or "ILS").upper()
    unit = "ש״ח" if currency == "ILS" else currency
    category = str(payload.get("suggested_category") or "לא ידוע")
    raw_reasons = payload.get("reasons")
    reasons = raw_reasons if isinstance(raw_reasons, list) else []
    reason_text = ", ".join(_REASON_TEXT.get(str(reason), str(reason)) for reason in reasons)
    money = f"{amount} {unit}".strip()
    lines = [
        str(payload.get("prompt_he") or "לאשר את שמירת ההוצאה?"),
        f"{merchant}: {money}",
        f"קטגוריה מוצעת: {category}",
    ]
    if reason_text:
        lines.append(f"צריך אישור כי {reason_text}.")
    lines.append("אפשר לאשר, לדחות, או לבחור קטגוריה אחרת.")
    return "\n".join(lines)


def _graph_input(user_id: int, message: str, receipt_path: str | Path | None) -> dict[str, object]:
    kwargs: dict[str, str] = {}
    if receipt_path is not None:
        kwargs["receipt_path"] = str(receipt_path)
    return {
        "messages": [HumanMessage(content=message, additional_kwargs=kwargs)],
        "user_id": user_id,
    }


def _turn_from_snapshot(
    user_id: int,
    thread_id: str,
    snapshot: object,
    result: object,
    trace: TurnTrace | None = None,
) -> ChatTurnResponse:
    payload = _interrupt_payload(snapshot)
    values = _values(snapshot)
    if not values and isinstance(result, dict):
        values = result
    if payload is not None:
        pending = values.get("pending_receipt")
        return ChatTurnResponse(
            status="interrupted",
            user_id=user_id,
            thread_id=thread_id,
            message=format_interrupt_message(payload),
            pending_receipt=pending if isinstance(pending, dict) else None,
            interrupt=dict(payload),
            trace=trace,
        )
    return ChatTurnResponse(
        status="completed",
        user_id=user_id,
        thread_id=thread_id,
        message=_final_message(values),
        pending_receipt=None,
        interrupt=None,
        trace=trace,
    )


def _interrupt_payload(snapshot: object) -> dict[str, Any] | None:
    raw = getattr(snapshot, "interrupts", ())
    if not raw:
        return None
    value = getattr(raw[0], "value", None)
    if isinstance(value, dict):
        return dict(value)
    return None


def _values(snapshot: object) -> dict[str, Any]:
    raw = getattr(snapshot, "values", None)
    if isinstance(raw, dict):
        return raw
    return {}


def _final_message(values: Mapping[str, Any]) -> str:
    text = values.get("final_response")
    if isinstance(text, str) and text.strip():
        return text.strip()
    messages = values.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if getattr(message, "type", None) != "ai":
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return "אין לי תשובה כרגע."
