"""Read-only figures for the financial overview."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func, select

from app.db.models import Category, Transaction
from app.db.session import get_session
from app.db.vector_store import get_user_preferences_collection

logger = logging.getLogger(__name__)

DEMO_USER_ID = 1
LOCAL_TZ = ZoneInfo("Asia/Jerusalem")
_MONTH_KEY = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def spending_kpis(user_id: int = DEMO_USER_ID) -> dict[str, Any]:
    """Total spend, transaction count, and the category with the highest total."""
    with get_session() as session:
        total = session.scalar(select(func.coalesce(func.sum(Transaction.amount), 0)).where(Transaction.user_id == user_id))
        count = session.scalar(
            select(func.count(Transaction.id)).where(Transaction.user_id == user_id)
        )
        top = session.execute(
            select(Category.name, func.sum(Transaction.amount))
            .join(Transaction, Transaction.category_id == Category.id)
            .where(Transaction.user_id == user_id)
            .group_by(Category.name)
            .order_by(func.sum(Transaction.amount).desc())
            .limit(1)
        ).first()

    top_name = "—"
    top_amount = 0.0
    if top is not None:
        top_name = str(top[0])
        top_amount = float(top[1] or 0)
    return {
        "total_spend": float(total or 0),
        "transaction_count": int(count or 0),
        "top_category": top_name,
        "top_category_amount": top_amount,
    }


def transaction_months(user_id: int = DEMO_USER_ID) -> list[str]:
    """Distinct local ``YYYY-MM`` months that have expenses, newest first."""
    with get_session() as session:
        stamps = session.scalars(
            select(Transaction.transacted_at).where(Transaction.user_id == user_id)
        ).all()
    months = {_month_key(stamp) for stamp in stamps if isinstance(stamp, datetime)}
    return sorted(months, reverse=True)


def spend_by_category(user_id: int = DEMO_USER_ID, period: str | None = None) -> pd.DataFrame:
    """Category totals ordered from largest to smallest.

    ``period`` is a ``YYYY-MM`` month in Asia/Jerusalem. ``None`` includes every expense.
    """
    selected = period if period is not None and _MONTH_KEY.fullmatch(period) else None
    with get_session() as session:
        rows = session.execute(
            select(Category.name, Transaction.amount, Transaction.transacted_at)
            .join(Transaction, Transaction.category_id == Category.id)
            .where(Transaction.user_id == user_id)
        ).all()

    totals: dict[str, float] = {}
    for name, amount, when in rows:
        if not isinstance(when, datetime):
            continue
        if selected is not None and _month_key(when) != selected:
            continue
        key = str(name)
        totals[key] = totals.get(key, 0.0) + float(amount or 0)

    ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return pd.DataFrame(
        [{"category": name, "total": total} for name, total in ordered],
        columns=["category", "total"],
    )


def latest_transactions(user_id: int = DEMO_USER_ID, limit: int = 8) -> pd.DataFrame:
    """Newest expenses for the live transactions table."""
    with get_session() as session:
        rows = session.execute(
            select(
                Transaction.transacted_at,
                Transaction.merchant,
                Category.name,
                Transaction.amount,
                Transaction.currency,
            )
            .join(Category, Category.id == Transaction.category_id)
            .where(Transaction.user_id == user_id)
            .order_by(Transaction.transacted_at.desc(), Transaction.id.desc())
            .limit(limit)
        ).all()

    records = [
        {
            "תאריך": _local_date(when),
            "בית עסק": merchant,
            "קטגוריה": category,
            "סכום": f"{float(amount):,.2f} {_currency_label(currency)}",
        }
        for when, merchant, category, amount, currency in rows
    ]
    return pd.DataFrame(records, columns=["תאריך", "בית עסק", "קטגוריה", "סכום"])


def category_names_by_slug(user_id: int = DEMO_USER_ID) -> dict[str, str]:
    """Hebrew category names keyed by the slug stored on preference rules."""
    try:
        with get_session() as session:
            rows = session.execute(
                select(Category.slug, Category.name).where(Category.user_id == user_id)
            ).all()
    except Exception:
        logger.exception("Could not read category names")
        return {}
    return {str(slug): str(name) for slug, name in rows}


def active_preference_rules() -> pd.DataFrame:
    """Documents currently stored in the Chroma user_preferences collection."""
    columns = ["בית עסק", "קטגוריה", "כלל"]
    try:
        collection = get_user_preferences_collection()
        if int(collection.count()) == 0:
            return pd.DataFrame(columns=columns)
        payload = collection.get(include=["documents", "metadatas"])
    except Exception:
        logger.exception("Could not read Chroma preference rules")
        return pd.DataFrame(columns=columns)

    documents = payload.get("documents") if isinstance(payload, dict) else None
    metadatas = payload.get("metadatas") if isinstance(payload, dict) else None
    docs = list(documents or [])
    metas = list(metadatas or [])
    records: list[dict[str, str]] = []
    for index, document in enumerate(docs):
        metadata = metas[index] if index < len(metas) and isinstance(metas[index], dict) else {}
        merchant = metadata.get("merchant") or metadata.get("topic") or "—"
        category = metadata.get("suggested_category") or metadata.get("category") or metadata.get("topic") or "—"
        records.append(
            {
                "בית עסק": str(merchant),
                "קטגוריה": str(category),
                "כלל": str(document or ""),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _month_key(value: datetime) -> str:
    return _local_date(value)[:7]


def _local_date(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(LOCAL_TZ).date().isoformat()


def _currency_label(currency: str) -> str:
    if str(currency).upper() == "ILS":
        return "₪"
    return str(currency)
