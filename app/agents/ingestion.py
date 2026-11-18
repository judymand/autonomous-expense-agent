"""Ingestion node, approval gate, and the commit that follows a decision."""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Category, Transaction, User
from app.db.receipt_hash import compute_receipt_hash
from app.graph.runtime import as_model, configurable, llm_from_config, preferences_from_config, session_scope
from app.graph.state import AgentState
from app.schemas.receipt import IngestionResult, ReceiptExtraction
from app.services.ingestion import find_duplicate, ingest_receipt, normalize_currency
from app.services.llm import get_chat_model

logger = logging.getLogger(__name__)

LOCAL_TZ = ZoneInfo("Asia/Jerusalem")
APPROVAL_AMOUNT_ILS = Decimal("500")
MIN_AUTO_CONFIDENCE = Decimal("0.80")
_RECEIPT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"}


def ingestion_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
    """Extract an expense and hold it until the approval gate."""
    user_text, receipt_file = _latest_expense_input(state)
    with session_scope(config) as session:
        if session.get(User, state["user_id"]) is None:
            raise LookupError(f"User {state['user_id']} does not exist")
        if receipt_file is not None:
            result = ingest_receipt(
                receipt_file,
                user_id=state["user_id"],
                session=session,
                llm=_model(config),
                preferences=_optional_preferences(config),
            )
        else:
            result = extract_text_expense(
                user_text,
                user_id=state["user_id"],
                session=session,
                llm=_model(config),
            )

    pending = pending_receipt(result)
    reasons = approval_reasons(pending)
    return {
        "pending_receipt": pending,
        "requires_approval": bool(reasons),
        "approval_status": "pending" if reasons else "auto_approved",
        "sql_result": None,
        "anomalies": [],
    }


def approval_gate(state: AgentState, config: RunnableConfig) -> dict[str, object]:
    """Pause before a risky write and continue with the user's decision.

    ``interrupt`` raises on the first visit and returns the resume value when
    the same node runs again. Safe expenses do not pause.
    """
    from langgraph.types import interrupt

    pending = dict(state.get("pending_receipt") or {})
    logger.info(
        "Approval gate for user %s thread %s",
        state.get("user_id"),
        configurable(config).get("thread_id"),
    )
    if not pending:
        return {
            "approval_status": "rejected",
            "final_response": "אין הוצאה שממתינה לאישור.",
        }

    reasons = approval_reasons(pending)
    if not reasons:
        return {"requires_approval": False, "approval_status": "auto_approved"}

    decision = interrupt(
        {
            "kind": "expense_approval",
            "reasons": reasons,
            "prompt_he": "לאשר את שמירת ההוצאה?",
            "merchant_name": pending.get("merchant_name"),
            "total_amount": pending.get("total_amount"),
            "currency": pending.get("currency"),
            "suggested_category": pending.get("suggested_category"),
            "confidence_score": pending.get("confidence_score"),
            "is_duplicate": bool(pending.get("is_duplicate")),
        }
    )
    approved, corrected = parse_approval(decision)
    if corrected:
        pending["corrected_category"] = corrected
    return {
        "pending_receipt": pending,
        "requires_approval": True,
        "approval_status": "approved" if approved else "rejected",
    }


def commit_expense_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
    """Write an approved expense and store a preference when the category changed."""
    pending = dict(state.get("pending_receipt") or {})
    status = state.get("approval_status")
    if status not in {"approved", "auto_approved"} or not pending:
        text = "ההוצאה לא נשמרה."
        return {"final_response": text, "messages": [AIMessage(content=text)]}

    category_name = str(pending.get("corrected_category") or pending.get("suggested_category") or "").strip()
    original = str(pending.get("original_category") or pending.get("suggested_category") or "").strip()
    corrected = str(pending.get("corrected_category") or "").strip()
    preference_saved = False

    with session_scope(config) as session:
        user = session.get(User, state["user_id"])
        if user is None:
            raise LookupError(f"User {state['user_id']} does not exist")
        if pending.get("is_duplicate"):
            text = "ההוצאה הזו כבר שמורה, אז לא שמרתי אותה שוב."
        else:
            category = category_for_name(session, user.id, category_name)
            session.add(_build_transaction(user.id, category.id, pending))
            session.flush()
            category_name = category.name
            text = (
                f"שמרתי הוצאה של {pending.get('total_amount')} ש״ח "
                f"אצל {pending.get('merchant_name')} בקטגוריה {category_name}."
            )
        if corrected and corrected != original:
            category = category_for_name(session, user.id, corrected)
            save_category_preference(
                preferences_from_config(config),
                user=user,
                merchant=str(pending.get("merchant_name") or ""),
                category_name=category.name,
                category_slug=category.slug,
            )
            preference_saved = True
            category_name = category.name
        session.commit()

    if preference_saved:
        text += f" מעכשיו אסווג את {pending.get('merchant_name')} כ{category_name}."
    return {"final_response": text, "messages": [AIMessage(content=text)]}


def extract_text_expense(
    text: str,
    *,
    user_id: int,
    session: Session,
    llm: object | None = None,
) -> IngestionResult:
    """Turn a free-text expense into the same shape as a receipt extraction."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("expense text is required")

    names = list(
        session.scalars(select(Category.name).where(Category.user_id == user_id).order_by(Category.name)).all()
    )
    categories = ", ".join(names) if names else "a short Hebrew spending category"
    prompt = (
        "Extract one expense from the text. "
        f"suggested_category must be one of: {categories}. "
        "total_amount is a positive number. currency is an ISO code such as ILS. "
        "confidence_score is your certainty from 0 to 1. "
        "Keep merchant_name in the original language.\n"
        "Expense text:\n"
        f"{cleaned}"
    )
    model = llm if llm is not None else get_chat_model()
    try:
        runnable = getattr(model, "with_structured_output")(ReceiptExtraction)
        parsed = getattr(runnable, "invoke")([HumanMessage(content=prompt)])
    except Exception:
        logger.exception("Failed to extract a text expense for user %s", user_id)
        raise

    extraction = as_model(parsed, ReceiptExtraction)
    extraction = extraction.model_copy(update={"currency": normalize_currency(extraction.currency)})
    digest = compute_receipt_hash(
        extraction.merchant_name,
        extraction.transaction_date,
        extraction.total_amount,
        extraction.currency,
    )
    matched = find_duplicate(session, user_id, digest)
    return IngestionResult(
        extraction=extraction,
        receipt_hash=digest,
        is_duplicate=matched is not None,
        matched_transaction_id=None if matched is None else matched.id,
        preference_applied=False,
    )


def pending_receipt(result: IngestionResult) -> dict[str, object]:
    """JSON-safe receipt held in graph state until commit."""
    extraction = result.extraction
    amount = Decimal(extraction.total_amount).quantize(Decimal("0.01"))
    payload: dict[str, object] = {
        "merchant_name": extraction.merchant_name,
        "transaction_date": extraction.transaction_date.isoformat(),
        "total_amount": f"{amount:.2f}",
        "currency": extraction.currency,
        "suggested_category": extraction.suggested_category,
        "original_category": extraction.suggested_category,
        "confidence_score": float(extraction.confidence_score),
        "receipt_hash": result.receipt_hash,
        "is_duplicate": result.is_duplicate,
        "matched_transaction_id": result.matched_transaction_id,
        "items": [item.name for item in extraction.items],
    }
    if result.matched_rule:
        payload["matched_rule"] = result.matched_rule
    return payload


def approval_reasons(pending: dict[str, object]) -> list[str]:
    """Return why this expense must be confirmed before it is stored."""
    reasons: list[str] = []
    currency = str(pending.get("currency") or "ILS").upper()
    amount = Decimal(str(pending.get("total_amount") or "0"))
    if currency == "ILS" and amount > APPROVAL_AMOUNT_ILS:
        reasons.append("amount_over_500")
    confidence = Decimal(str(pending.get("confidence_score") or "0"))
    if confidence < MIN_AUTO_CONFIDENCE:
        reasons.append("low_confidence")
    if bool(pending.get("is_duplicate")):
        reasons.append("duplicate")
    return reasons


def parse_approval(value: object) -> tuple[bool, str | None]:
    """Read an approval flag and an optional corrected category name."""
    if isinstance(value, bool):
        return value, None
    if isinstance(value, str):
        approved = value.strip().casefold() in {"approve", "approved", "yes", "כן", "אישור", "מאשר", "אשרו"}
        return approved, None
    if isinstance(value, dict):
        raw_category = value.get("category", value.get("corrected_category"))
        category = raw_category.strip() if isinstance(raw_category, str) and raw_category.strip() else None
        if "approved" in value:
            return bool(value["approved"]), category
        if "approve" in value:
            return bool(value["approve"]), category
        return category is not None, category
    raise ValueError("Approval response must be a boolean, text, or object")


def category_for_name(session: Session, user_id: int, name: str) -> Category:
    """Return the user's category with this Hebrew name, creating it when needed."""
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("category name is required")
    found = session.scalar(select(Category).where(Category.user_id == user_id, Category.name == cleaned))
    if found is not None:
        return found
    slug = "rule-" + hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
    created = Category(user_id=user_id, slug=slug, name=cleaned)
    session.add(created)
    session.flush()
    return created


def save_category_preference(
    collection: object,
    *,
    user: User,
    merchant: str,
    category_name: str,
    category_slug: str,
) -> None:
    """Upsert a merchant-specific rule into the user_preferences collection."""
    document = f"תמיד לסווג עסקאות של {merchant} בקטגוריה {category_name}."
    digest = hashlib.sha256(f"{user.id}|{merchant.casefold()}|{category_name}".encode("utf-8")).hexdigest()
    metadata: dict[str, str | int] = {
        "telegram_id": int(user.telegram_id or 0),
        "merchant": merchant,
        "suggested_category": category_name,
        "topic": category_slug,
        "language": "he",
    }
    try:
        upsert = getattr(collection, "upsert")
        upsert(
            ids=[f"pref-{digest[:24]}"],
            documents=[document],
            metadatas=[metadata],
        )
    except Exception:
        logger.exception("Failed to save a category preference for user %s", user.id)
        raise
    logger.info("Saved category preference for %s as %s", merchant, category_name)


def _build_transaction(user_id: int, category_id: int, pending: dict[str, object]) -> Transaction:
    day = date.fromisoformat(str(pending["transaction_date"]))
    local_noon = datetime(day.year, day.month, day.day, 12, 0, tzinfo=LOCAL_TZ)
    items = pending.get("items")
    description = None
    if isinstance(items, list) and items:
        description = ", ".join(str(item) for item in items)[:255]
    return Transaction(
        user_id=user_id,
        category_id=category_id,
        merchant=str(pending["merchant_name"]),
        description=description,
        amount=Decimal(str(pending["total_amount"])),
        currency=str(pending.get("currency") or "ILS"),
        receipt_hash=str(pending["receipt_hash"]),
        transacted_at=local_noon.astimezone(timezone.utc),
    )


def _model(config: RunnableConfig) -> object:
    model = llm_from_config(config)
    if model is None:
        return get_chat_model()
    return model


def _optional_preferences(config: RunnableConfig) -> object | None:
    try:
        return preferences_from_config(config)
    except Exception:
        logger.exception("Preference collection is unavailable during ingestion")
        return None


def _latest_expense_input(state: AgentState) -> tuple[str, Path | None]:
    messages = state.get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", None) not in {None, "human"}:
            continue
        content = getattr(message, "content", "")
        kwargs = getattr(message, "additional_kwargs", {}) or {}
        raw_path = kwargs.get("receipt_path") if isinstance(kwargs, dict) else None
        if isinstance(raw_path, str) and Path(raw_path).is_file():
            return _content_text(content), Path(raw_path)
        if isinstance(content, str):
            candidate = Path(content.strip())
            if content.strip() and candidate.suffix.lower() in _RECEIPT_SUFFIXES and candidate.is_file():
                return content, candidate
            return content, None
        return _content_text(content), None
    return "", None


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)
