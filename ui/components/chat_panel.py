"""Hebrew chat transcript and the human approval card."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any, TypeGuard

import streamlit as st
from streamlit.typing import ChatInputValue

SendHandler = Callable[[str, tuple[str, bytes] | None], None]
ResolveHandler = Callable[[bool, str | None], None]


def render_chat_panel(
    *,
    history: list[dict[str, str]],
    pending: dict[str, Any] | None,
    categories: list[str],
    error: str | None,
    on_send: SendHandler,
    on_resolve: ResolveHandler,
) -> None:
    """Render the transcript, the approval card, and the composer."""
    if error:
        st.error(error)

    queued = st.session_state.get("pending_turn")
    busy = isinstance(queued, dict)
    with st.container(height=460, key="chat_transcript"):
        if not history:
            st.markdown(
                _bubble("assistant", "אפשר לשלוח קבלה, לתאר הוצאה, או לשאול על התקציב."),
                unsafe_allow_html=True,
            )
        for item in history:
            role = "user" if item.get("role") == "user" else "assistant"
            content = html.escape(item.get("content", "")).replace("\n", "<br>")
            st.markdown(_bubble(role, content), unsafe_allow_html=True)
        assistant_slot = st.empty()

    if pending is not None and not busy:
        _render_approval(pending, categories, on_resolve)
        return

    submission = st.chat_input(
        "כתבו הודעה או צרפו קבלה...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "pdf"],
        disabled=busy,
        key="chat_composer",
    )

    if busy:
        message = str(queued.get("message") or "")
        upload = queued.get("upload")
        file_payload = upload if _is_upload(upload) else None
        with assistant_slot.container():
            with st.spinner("מפעיל סוכנים ומעבד פנייה..."):
                on_send(message, file_payload)
        st.session_state.pending_turn = None
        st.rerun()

    if isinstance(submission, ChatInputValue):
        _queue_turn(submission)


def _queue_turn(submission: ChatInputValue) -> None:
    """Phase 1: show the user bubble immediately, then rerun into the agent call."""
    text = submission.text.strip()
    upload = _first_upload(submission)
    if not text and upload is None:
        return
    label = text or "קבלה מצורפת"
    if upload is not None:
        label = f"{label}\n📎 {upload[0]}"
    history = list(st.session_state.get("chat_history") or [])
    history.append({"role": "user", "content": label})
    st.session_state.chat_history = history
    st.session_state.pending_turn = {"message": text, "upload": upload}
    st.rerun()


def _first_upload(submission: ChatInputValue) -> tuple[str, bytes] | None:
    files = submission.files
    if not files:
        return None
    uploaded = files[0]
    payload = uploaded.getvalue()
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    return str(uploaded.name), payload


def _is_upload(value: object) -> TypeGuard[tuple[str, bytes]]:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], bytes)
    )


def _bubble(role: str, content: str) -> str:
    """One transcript turn, with a small assistant label on replies."""
    label = '<div class="bubble-label">🤖 Assistant</div>' if role == "assistant" else ""
    return f'<div class="chat-log"><div class="bubble {role}">{label}{content}</div></div>'


def _render_approval(
    pending: dict[str, Any],
    categories: list[str],
    on_resolve: ResolveHandler,
) -> None:
    merchant = str(pending.get("merchant_name") or "העסק")
    amount = str(pending.get("total_amount") or "").strip()
    currency = str(pending.get("currency") or "ILS").upper()
    unit = "₪" if currency == "ILS" else currency
    category = str(pending.get("suggested_category") or "לא ידוע")
    confidence = pending.get("confidence_score")
    confidence_text = f"{float(confidence):.0%}" if isinstance(confidence, int | float) else "—"

    st.markdown(
        (
            '<div class="approval-card" dir="rtl">'
            "<div class='approval-kicker'>ממתין לאישור</div>"
            f"<div class='approval-merchant'>{html.escape(merchant)}</div>"
            f"<div class='approval-amount'>{html.escape(amount)} {html.escape(unit)}</div>"
            f"<div class='approval-meta'>קטגוריה מוצעת: {html.escape(category)}</div>"
            f"<div class='approval-meta'>ביטחון: {html.escape(confidence_text)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    approve, reject = st.columns(2)
    if approve.button("Approve ✅", key="approve_expense", width="stretch"):
        on_resolve(True, None)
    if reject.button("Reject ❌", key="reject_expense", width="stretch"):
        on_resolve(False, None)

    if categories:
        choice = st.selectbox("Override Category", categories, key="override_category")
    else:
        choice = st.text_input("Override Category", key="override_category_text")
    if st.button("Override Category", width="stretch"):
        cleaned = str(choice).strip()
        if cleaned:
            on_resolve(True, cleaned)
