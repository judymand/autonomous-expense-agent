"""Streamlit entry point for the autonomous expense assistant.

The page calls the LangGraph runner in-process. It does not start FastAPI.
On a cold Streamlit Cloud boot it creates the SQLite file and seeds the demo
user when the users table is empty.
"""

from __future__ import annotations

import logging
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.services.assistant import AssistantError
from app.services.llm import prime_llm_environment
from ui.components.chat_panel import render_chat_panel
from ui.components.dashboard_panel import render_kpi_row
from ui.components.inspector_panel import render_inspector_panel
from ui.services.agent_gateway import latest_extraction, load_categories, resolve_turn, send_turn
from ui.styles import APP_CSS

logger = logging.getLogger(__name__)
_STARTUP_LOCK = threading.Lock()


def main() -> None:
    """Configure the page, boot storage, and render the two columns."""
    st.set_page_config(
        page_title="עוזר ההוצאות",
        page_icon="💸",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    try:
        prime_llm_environment()
        _ensure_demo_database()
    except Exception:
        logger.exception("Streamlit startup failed")
        st.error("המערכת לא עלתה. בדקו את המפתח ואת קבצי הנתונים.")
        st.stop()

    _ensure_session()
    _render_header()
    error = st.session_state.pop("form_error", None)
    categories = _safe_categories()

    insight_col, chat_col = st.columns([0.85, 1.15], gap="large")
    with chat_col:
        render_chat_panel(
            history=list(st.session_state.chat_history),
            pending=st.session_state.pending_interrupt,
            categories=categories,
            error=error if isinstance(error, str) else None,
            on_send=_on_send,
            on_resolve=_on_resolve,
        )
    with insight_col:
        render_inspector_panel(
            trace=st.session_state.last_trace,
            extraction=st.session_state.last_extraction,
        )


def _ensure_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:16]}"
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_trace", None)
    st.session_state.setdefault("last_extraction", None)
    st.session_state.setdefault("pending_interrupt", None)
    st.session_state.setdefault("upload_key", "receipt-0")


def _render_header() -> None:
    action_col, title_col = st.columns([1, 5])
    with title_col:
        st.markdown(
            (
                '<div class="hero">'
                "<h1>עוזר ההוצאות</h1>"
                '<span class="badge">מערכת סוכנים אוטונומית לניהול הוצאות</span>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with action_col:
        if st.button("שיחה חדשה", width="stretch"):
            _reset_thread()
            st.rerun()
    render_kpi_row()


def _reset_thread() -> None:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:16]}"
    st.session_state.chat_history = []
    st.session_state.last_trace = None
    st.session_state.last_extraction = None
    st.session_state.pending_interrupt = None
    st.session_state.upload_key = f"receipt-{uuid.uuid4().hex[:8]}"


def _safe_categories() -> list[str]:
    try:
        return load_categories()
    except Exception:
        logger.exception("Could not load categories")
        return []


def _on_send(message: str, upload: tuple[str, bytes] | None) -> None:
    text = message.strip()
    if not text and upload is None:
        return
    label = text or "קבלה מצורפת"
    if upload is not None:
        label = f"{label}\n📎 {upload[0]}"
    try:
        turn = send_turn(thread_id=str(st.session_state.thread_id), message=text, upload=upload)
    except AssistantError as exc:
        st.session_state.form_error = exc.detail
        st.rerun()
    except ValueError as exc:
        st.session_state.form_error = str(exc)
        st.rerun()
    except Exception:
        logger.exception("UI chat turn failed")
        st.session_state.form_error = "לא הצלחתי להשלים את הפנייה."
        st.rerun()

    history = list(st.session_state.chat_history)
    history.append({"role": "user", "content": label})
    history.append({"role": "assistant", "content": turn.message})
    st.session_state.chat_history = history
    _remember_turn(turn)
    st.session_state.upload_key = f"receipt-{uuid.uuid4().hex[:8]}"
    st.rerun()


def _on_resolve(approved: bool, category: str | None) -> None:
    try:
        turn = resolve_turn(
            thread_id=str(st.session_state.thread_id),
            approved=approved,
            category=category,
        )
    except AssistantError as exc:
        st.session_state.form_error = exc.detail
        st.rerun()
    except Exception:
        logger.exception("UI resume failed")
        st.session_state.form_error = "לא הצלחתי לשמור את ההחלטה."
        st.rerun()

    history = list(st.session_state.chat_history)
    history.append({"role": "assistant", "content": turn.message})
    st.session_state.chat_history = history
    _remember_turn(turn)
    st.rerun()


def _remember_turn(turn: Any) -> None:
    trace = turn.trace.model_dump(mode="json") if turn.trace is not None else None
    st.session_state.last_trace = trace
    visited = list(trace.get("visited_nodes", [])) if isinstance(trace, dict) else []
    if turn.pending_receipt:
        st.session_state.last_extraction = dict(turn.pending_receipt)
    elif any(name in visited for name in ("ingestion", "approval_gate", "commit_expense")):
        st.session_state.last_extraction = latest_extraction(str(st.session_state.thread_id))
    else:
        st.session_state.last_extraction = None

    if turn.status == "interrupted":
        st.session_state.pending_interrupt = _pending_view(turn)
    else:
        st.session_state.pending_interrupt = None


def _pending_view(turn: Any) -> dict[str, Any]:
    receipt = dict(turn.pending_receipt or {})
    interrupt = dict(turn.interrupt or {})
    confidence = receipt.get("confidence_score", interrupt.get("confidence_score"))
    return {
        "message": turn.message,
        "merchant_name": interrupt.get("merchant_name") or receipt.get("merchant_name"),
        "total_amount": interrupt.get("total_amount") or receipt.get("total_amount"),
        "currency": interrupt.get("currency") or receipt.get("currency") or "ILS",
        "suggested_category": interrupt.get("suggested_category") or receipt.get("suggested_category"),
        "confidence_score": confidence,
    }


def _ensure_demo_database() -> None:
    """Create tables and seed user 1 when the database has no users yet."""
    from sqlalchemy import func, select

    from app.db.models import User
    from app.db.seed import seed_demo_data
    from app.db.session import get_session, init_db

    with _STARTUP_LOCK:
        init_db()
        with get_session() as session:
            count = int(session.scalar(select(func.count()).select_from(User)) or 0)
        if count == 0:
            logger.info("Users table is empty; seeding demo data")
            seed_demo_data()


main()
