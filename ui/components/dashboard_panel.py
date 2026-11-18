"""Financial overview backed by live SQLite reads."""

from __future__ import annotations

import logging

import streamlit as st

from ui.services.analytics_queries import latest_transactions, spend_by_category, spending_kpis

logger = logging.getLogger(__name__)


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
        breakdown = spend_by_category()
        recent = latest_transactions()
    except Exception:
        logger.exception("Dashboard query failed")
        st.error("לא הצלחתי לקרוא את ההוצאות.")
        return

    st.markdown('<div class="section-label">הוצאות לפי קטגוריה</div>', unsafe_allow_html=True)
    if breakdown.empty:
        st.caption("אין עדיין הוצאות.")
    else:
        st.bar_chart(breakdown, x="category", y="total", horizontal=True, height=280)

    st.markdown('<div class="section-label">עסקאות אחרונות</div>', unsafe_allow_html=True)
    if recent.empty:
        st.caption("אין תנועות להצגה.")
    else:
        st.dataframe(recent, hide_index=True, width="stretch")
