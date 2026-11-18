"""Financial overview backed by live SQLite reads."""

from __future__ import annotations

import html
import logging

import streamlit as st

from ui.services.analytics_queries import (
    latest_transactions,
    spend_by_category,
    spending_kpis,
    transaction_months,
)

logger = logging.getLogger(__name__)

ALL_PERIOD = "כל התקופה"


def render_kpi_row() -> None:
    """Three headline figures across the top of the page."""
    try:
        kpis = spending_kpis()
    except Exception:
        logger.exception("KPI query failed")
        st.error("לא הצלחתי לקרוא את ההוצאות.")
        return

    total, count, top = st.columns(3)
    total.metric("סה״כ הוצאות", f"₪ {kpis['total_spend']:,.2f}")
    count.metric("מספר עסקאות", f"{kpis['transaction_count']:,}")
    top.metric("קטגוריה מובילה", str(kpis["top_category"]), f"₪ {kpis['top_category_amount']:,.0f}")


def render_spend_panel() -> None:
    """Category chart and the newest transactions for the expenses tab."""
    try:
        months = transaction_months()
    except Exception:
        logger.exception("Dashboard query failed")
        st.error("לא הצלחתי לקרוא את ההוצאות.")
        return

    period = _render_period_filter(months)
    title = _chart_title(period)
    selected = None if period == ALL_PERIOD else period
    try:
        breakdown = spend_by_category(period=selected)
        recent = latest_transactions()
    except Exception:
        logger.exception("Dashboard query failed")
        st.error("לא הצלחתי לקרוא את ההוצאות.")
        return

    if breakdown.empty:
        st.caption("אין עדיין הוצאות.")
    else:
        st.bar_chart(
            breakdown,
            x="category",
            y="total",
            horizontal=True,
            height=280,
            alt=title,
        )

    st.markdown('<div class="section-label">עסקאות אחרונות</div>', unsafe_allow_html=True)
    if recent.empty:
        st.caption("אין תנועות להצגה.")
    else:
        st.dataframe(recent, hide_index=True, width="stretch", alt="עסקאות אחרונות")


def _render_period_filter(months: list[str]) -> str:
    """Compact month picker. The chart title repeats the selected timeframe."""
    options = [ALL_PERIOD, *months]
    with st.container(key="spend_heading"):
        title_col, filter_col = st.columns([0.68, 0.32], vertical_alignment="bottom")
        period = str(filter_col.selectbox("תקופה", options, key="expense_period"))
        title_col.markdown(
            f'<div class="section-label">{html.escape(_chart_title(period))}</div>',
            unsafe_allow_html=True,
        )
    return period


def _chart_title(period: str) -> str:
    label = ALL_PERIOD if period == ALL_PERIOD else period
    return f"התפלגות הוצאות לפי קטגוריה ({label})"
