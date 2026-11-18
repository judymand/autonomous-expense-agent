"""Behind-the-scenes view of the multi-agent turn."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from ui.components.dashboard_panel import render_spend_panel
from ui.services.analytics_queries import active_preference_rules, category_names_by_slug

_EXPENSE_ROUTE: tuple[tuple[str, str], ...] = (
    ("supervisor", "Supervisor"),
    ("ingestion", "Ingestion"),
    ("approval_gate", "Approval Gate"),
    ("commit_expense", "SQLite"),
)
_ANALYSIS_ROUTE: tuple[tuple[str, str], ...] = (
    ("supervisor", "Supervisor"),
    ("analyst", "Analyst"),
    ("advisor", "Advisor"),
)
_OTHER_ROUTES: dict[str, tuple[tuple[str, str], ...]] = {
    "preference_updater": (("supervisor", "Supervisor"), ("preference_updater", "Preferences")),
    "general_chat": (("supervisor", "Supervisor"), ("general_chat", "General Chat")),
}


def render_inspector_panel(
    *,
    trace: dict[str, Any] | None,
    extraction: dict[str, Any] | None,
) -> None:
    """Show the route, the receipt, the SQL, and the spend chart in separate tabs."""
    st.markdown('<div class="panel-title">מאחורי הקלעים</div>', unsafe_allow_html=True)
    visited = _visited(trace)
    visible = _extraction_view(extraction)
    _render_pipeline(visited)
    route_tab, sql_tab, chart_tab = st.tabs(["ניתוב ונתוני קבלה", "שאילתת SQL וחוקים", "גרף הוצאות"])

    with route_tab:
        st.markdown('<div class="section-label">נתוני קבלה</div>', unsafe_allow_html=True)
        if visible:
            st.json(visible, expanded=True)
        else:
            _empty_state("אין חילוץ בפנייה הזו")
        st.markdown('<div class="section-label">ביטחון</div>', unsafe_allow_html=True)
        _render_gauge(visible.get("confidence_score") if visible else None)

    with sql_tab:
        st.markdown('<div class="section-label">שאילתה</div>', unsafe_allow_html=True)
        statement = None if trace is None else trace.get("generated_sql")
        if isinstance(statement, str) and statement.strip():
            st.code(statement.strip(), language="sql")
        else:
            _empty_state("לא הורצה שאילתה בפנייה הזו")
        st.markdown('<div class="section-label">כלל שהוחל</div>', unsafe_allow_html=True)
        rule = None if trace is None else trace.get("matched_rule")
        if isinstance(rule, dict) and rule:
            st.json(rule, expanded=False)
        else:
            _empty_state("לא הוחל כלל סיווג בפנייה הזו")
        st.markdown('<div class="section-label">כללי ChromaDB</div>', unsafe_allow_html=True)
        _render_preference_rules()

    with chart_tab:
        render_spend_panel()


def _render_preference_rules() -> None:
    """Full-width Hebrew cards for each stored preference rule."""
    rules = active_preference_rules()
    if rules.empty:
        _empty_state("אין כללים שמורים")
        return
    labels = category_names_by_slug()
    cards: list[str] = []
    for record in rules.to_dict(orient="records"):
        raw_category = str(record.get("קטגוריה") or "—")
        category = html.escape(labels.get(raw_category, raw_category))
        merchant = html.escape(str(record.get("בית עסק") or "—"))
        body = html.escape(str(record.get("כלל") or "")).replace("\n", "<br>")
        cards.append(
            '<article class="rule-card" dir="rtl">'
            '<div class="rule-top">'
            f'<span class="rule-badge">{category}</span>'
            f'<span class="rule-merchant">{merchant}</span>'
            "</div>"
            f'<p class="rule-text">{body}</p>'
            "</article>"
        )
    st.markdown(f'<div class="rule-list">{"".join(cards)}</div>', unsafe_allow_html=True)


def _empty_state(message: str) -> None:
    """Centered dashed card for a section that has nothing to show."""
    st.markdown(f'<div class="empty-state">{html.escape(message)}</div>', unsafe_allow_html=True)


def _visited(trace: dict[str, Any] | None) -> list[str]:
    if not trace:
        return []
    raw = trace.get("visited_nodes")
    if not isinstance(raw, list):
        return []
    return [str(name) for name in raw]


def _extraction_view(extraction: dict[str, Any] | None) -> dict[str, Any]:
    if not extraction:
        return {}
    hidden = {"receipt_hash", "matched_rule"}
    return {key: value for key, value in extraction.items() if key not in hidden}


def _render_pipeline(visited: list[str]) -> None:
    active = set(visited)
    routes = _routes_for(active)
    blocks: list[str] = []
    for route in routes:
        parts: list[str] = []
        current = visited[-1] if visited else None
        for index, (node_id, label) in enumerate(route):
            if index:
                parts.append('<span class="pipe-arrow">→</span>')
            if node_id == current:
                css = "node now"
            elif node_id in active:
                css = "node done"
            else:
                css = "node"
            parts.append(f'<span class="{css}">{html.escape(label)}</span>')
        blocks.append(f'<div class="pipeline">{"".join(parts)}</div>')
    st.markdown("".join(blocks), unsafe_allow_html=True)


def _routes_for(active: set[str]) -> list[tuple[tuple[str, str], ...]]:
    if not active:
        return [_EXPENSE_ROUTE, _ANALYSIS_ROUTE]
    if active.intersection({"ingestion", "approval_gate", "commit_expense"}):
        return [_EXPENSE_ROUTE]
    if active.intersection({"analyst", "advisor"}):
        return [_ANALYSIS_ROUTE]
    for node_id, route in _OTHER_ROUTES.items():
        if node_id in active:
            return [route]
    return [_EXPENSE_ROUTE, _ANALYSIS_ROUTE]


def _render_gauge(score: object) -> None:
    if not isinstance(score, int | float):
        st.caption("—")
        return
    value = max(0.0, min(1.0, float(score)))
    tone = "low" if value < 0.8 else "high"
    st.markdown(
        (
            '<div class="gauge">'
            f'<div class="gauge-fill {tone}" style="width: {value * 100:.1f}%"></div>'
            "</div>"
            f'<div class="gauge-label">{value:.0%}</div>'
        ),
        unsafe_allow_html=True,
    )
