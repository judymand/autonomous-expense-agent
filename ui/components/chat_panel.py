"""Hebrew chat transcript and the human approval card."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

import streamlit as st

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
    st.markdown('<div class="panel-title">השיחה</div>', unsafe_allow_html=True)
    if error:
        st.error(error)

    with st.container(height=460):
        if not history:
            st.markdown(
                '<div class="chat-log"><div class="bubble assistant">אפשר לשלוח קבלה, לתאר הוצאה, או לשאול על התקציב.</div></div>',
                unsafe_allow_html=True,
            )
        for item in history:
            role = "user" if item.get("role") == "user" else "assistant"
            content = html.escape(item.get("content", "")).replace("\n", "<br>")
            st.markdown(
                f'<div class="chat-log"><div class="bubble {role}">{content}</div></div>',
                unsafe_allow_html=True,
            )

    if pending is not None:
        _render_approval(pending, categories, on_resolve)
        return

    upload_key = str(st.session_state.get("upload_key", "receipt-0"))
    uploaded = st.file_uploader(
        "הוסף קבלה או חשבונית",
        type=["png", "jpg", "jpeg", "pdf"],
        key=upload_key,
    )
    if uploaded is not None and st.button("שליחת קבלה", type="primary"):
        on_send("", (uploaded.name, uploaded.getvalue()))
        return
    message = st.chat_input("כתבו הודעה או צרפו קבלה…")
    if message is None:
        return
    payload: tuple[str, bytes] | None = None
    if uploaded is not None:
        payload = (uploaded.name, uploaded.getvalue())
    on_send(message, payload)


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
