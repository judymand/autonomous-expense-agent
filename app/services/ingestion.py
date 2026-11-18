"""Extract a receipt with a vision model, apply preference rules, and detect duplicates."""

from __future__ import annotations

import base64
import logging
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Protocol

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Category, Transaction, User
from app.db.receipt_hash import compute_receipt_hash
from app.db.session import ensure_receipt_hash_column
from app.db.vector_store import get_user_preferences_collection
from app.schemas.receipt import IngestionResult, ReceiptExtraction
from app.services.llm import get_chat_model

logger = logging.getLogger(__name__)

MAX_PDF_PAGES = 4
# Cosine distance in Chroma (lower is closer). 0.40 keeps a real merchant near its
# own rule and rejects weaker hits, including boilerplate names such as "שם החברה בע\"מ".
MAX_PREFERENCE_DISTANCE = 0.40

_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_CURRENCY_ALIASES = {
    "ils": "ILS",
    "nis": "ILS",
    "₪": "ILS",
    "shekel": "ILS",
    "shekels": "ILS",
    "שקל": "ILS",
    'ש"ח': "ILS",
    "ש״ח": "ILS",
    "שח": "ILS",
}

# Invoice templates often leave this placeholder instead of the real merchant.
_GENERIC_MERCHANT_NAMES = frozenset({"שם החברה בעמ", "שם החברה", "שם העסק בעמ", "שם העסק"})


class SupportsStructuredOutput(Protocol):
    """Chat model that can return a Pydantic object."""

    def with_structured_output(self, schema: type[BaseModel]) -> object:
        """Return a runnable that parses model output into ``schema``."""


class PreferenceCollection(Protocol):
    """Subset of the Chroma collection API used for preference rules."""

    def count(self) -> int:
        """Return the number of stored preference documents."""

    def query(self, **kwargs: object) -> dict[str, object]:
        """Return the nearest preference documents."""


def ingest_receipt(
    source: Path,
    *,
    user_id: int,
    session: Session,
    llm: SupportsStructuredOutput | None = None,
    preferences: PreferenceCollection | None = None,
) -> IngestionResult:
    """Read a receipt file and return structured fields, category overrides, and duplicates."""
    user = session.get(User, user_id)
    if user is None:
        raise LookupError(f"User {user_id} does not exist")

    parts = load_vision_parts(source)
    category_names = _category_names(session, user_id)
    model = llm if llm is not None else get_chat_model()
    extraction = _extract(model, parts, category_names)
    extraction = extraction.model_copy(update={"currency": normalize_currency(extraction.currency)})

    preference_applied = False
    matched_rule: dict[str, object] | None = None
    applied = _apply_preference(extraction, user, session, preferences)
    if applied is not None:
        extraction, matched_rule = applied
        preference_applied = True

    digest = compute_receipt_hash(
        extraction.merchant_name,
        extraction.transaction_date,
        extraction.total_amount,
        extraction.currency,
    )
    matched = find_duplicate(session, user_id, digest)
    if matched is not None:
        logger.info("Duplicate receipt %s matches transaction %s", digest, matched.id)

    return IngestionResult(
        extraction=extraction,
        receipt_hash=digest,
        is_duplicate=matched is not None,
        matched_transaction_id=None if matched is None else matched.id,
        preference_applied=preference_applied,
        matched_rule=matched_rule,
    )


def load_vision_parts(source: Path) -> list[tuple[bytes, str]]:
    """Load an image as-is, or render each PDF page to PNG."""
    if not source.is_file():
        raise FileNotFoundError(source)

    suffix = source.suffix.lower()
    try:
        payload = source.read_bytes()
    except OSError:
        logger.exception("Could not read receipt file %s", source)
        raise

    if suffix == ".pdf":
        return [(page, "image/png") for page in render_pdf_pages(payload)]

    mime_type = _IMAGE_MIME_TYPES.get(suffix)
    if mime_type is None:
        raise ValueError(f"Unsupported receipt file type: {suffix or source.name}")
    return [(payload, mime_type)]


def render_pdf_pages(payload: bytes, *, limit: int = MAX_PDF_PAGES) -> list[bytes]:
    """Rasterize up to ``limit`` PDF pages so every vision provider receives images."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(payload)
    pages: list[bytes] = []
    try:
        page_count = len(document)
        if page_count == 0:
            raise ValueError("PDF did not contain any pages")
        if page_count > limit:
            logger.info("Using the first %s of %s PDF pages", limit, page_count)
        for index in range(min(page_count, limit)):
            page = document[index]
            bitmap = page.render(scale=2)
            image = bitmap.to_pil()
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            pages.append(buffer.getvalue())
            bitmap.close()
            page.close()
    except ValueError:
        raise
    except Exception:
        logger.exception("Failed to render a PDF receipt")
        raise
    finally:
        document.close()
    return pages


def normalize_currency(value: str) -> str:
    """Map shekel symbols and nicknames onto the ISO code used in the database."""
    cleaned = value.strip()
    return _CURRENCY_ALIASES.get(cleaned.casefold(), cleaned.upper())


def find_duplicate(session: Session, user_id: int, digest: str) -> Transaction | None:
    """Return an existing transaction with the same receipt hash for this user."""
    ensure_receipt_hash_column(session.connection())
    matched = session.scalar(
        select(Transaction).where(
            Transaction.user_id == user_id,
            Transaction.receipt_hash == digest,
        )
    )
    if matched is not None:
        return matched

    unhashed = session.scalars(
        select(Transaction).where(
            Transaction.user_id == user_id,
            Transaction.receipt_hash.is_(None),
        )
    ).all()
    for row in unhashed:
        row_digest = compute_receipt_hash(row.merchant, row.transacted_at, Decimal(str(row.amount)), row.currency)
        if row_digest != digest:
            continue
        row.receipt_hash = row_digest
        return row
    return None


def _extract(
    llm: SupportsStructuredOutput,
    parts: list[tuple[bytes, str]],
    category_names: list[str],
) -> ReceiptExtraction:
    try:
        runnable = llm.with_structured_output(ReceiptExtraction)
        invoke = getattr(runnable, "invoke")
        parsed = invoke([HumanMessage(content=_message_content(parts, category_names))])
    except Exception:
        logger.exception("Vision model failed to extract the receipt")
        raise

    if isinstance(parsed, ReceiptExtraction):
        return parsed
    return ReceiptExtraction.model_validate(parsed)


def _message_content(
    parts: list[tuple[bytes, str]],
    category_names: list[str],
) -> list[dict[str, object]]:
    categories = ", ".join(category_names) if category_names else "a short Hebrew spending category"
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Extract the purchase shown in this receipt. "
                "Keep merchant_name and item names in the receipt language, including Hebrew. "
                f"suggested_category must be one of: {categories}. "
                "transaction_date is the purchase date. "
                "total_amount is the final amount paid. "
                "currency is an ISO code such as ILS. "
                "confidence_score is your certainty from 0 to 1."
            ),
        }
    ]
    for payload, mime_type in parts:
        encoded = base64.b64encode(payload).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )
    return content


def _category_names(session: Session, user_id: int) -> list[str]:
    return list(
        session.scalars(
            select(Category.name).where(Category.user_id == user_id).order_by(Category.name)
        ).all()
    )


def _apply_preference(
    extraction: ReceiptExtraction,
    user: User,
    session: Session,
    preferences: PreferenceCollection | None,
) -> tuple[ReceiptExtraction, dict[str, object]] | None:
    if _is_generic_merchant(extraction.merchant_name):
        logger.info("Skipping preference rules for generic merchant %s", extraction.merchant_name)
        return None

    collection = preferences if preferences is not None else _open_preferences()
    if collection is None:
        return None

    try:
        result_count = int(collection.count())
    except Exception:
        logger.exception("Could not read user preference rules")
        return None
    if result_count <= 0:
        return None

    query = " ".join(
        [
            extraction.merchant_name,
            extraction.suggested_category,
            *[item.name for item in extraction.items],
        ]
    )
    where = {"telegram_id": user.telegram_id} if user.telegram_id is not None else None
    try:
        found = collection.query(
            query_texts=[query],
            n_results=min(4, result_count),
            where=where,
            include=["metadatas", "documents", "distances"],
        )
    except Exception:
        logger.exception("Preference lookup failed for merchant %s", extraction.merchant_name)
        return None

    categories_by_slug = {
        category.slug: category.name
        for category in session.scalars(select(Category).where(Category.user_id == user.id)).all()
    }
    category, index = _matching_preference(extraction.merchant_name, found, categories_by_slug)
    if category is None or index is None:
        return None

    rule = _rule_payload(found, index, category)
    if category == extraction.suggested_category:
        return extraction.model_copy(update={"confidence_score": 1.0}), rule

    logger.info("Preference rule set category %s for %s", category, extraction.merchant_name)
    return extraction.model_copy(update={"suggested_category": category, "confidence_score": 1.0}), rule


def _open_preferences() -> PreferenceCollection | None:
    try:
        return get_user_preferences_collection()
    except Exception:
        logger.exception("Could not open the user_preferences collection")
        return None


def _matching_preference(
    merchant: str,
    result: dict[str, object],
    categories_by_slug: dict[str, str],
) -> tuple[str | None, int | None]:
    """Return the winning category and its row index, using the same ranking as before."""
    metadatas = _first_list(result.get("metadatas"))
    distances = _first_list(result.get("distances"))
    best_category: str | None = None
    best_index: int | None = None
    best_rank: tuple[int, float] | None = None

    for index, metadata in enumerate(metadatas):
        if not isinstance(metadata, dict):
            continue
        category = _category_from_metadata(metadata, categories_by_slug)
        if category is None:
            continue
        distance = _distance_at(distances, index)
        if distance is None or distance > MAX_PREFERENCE_DISTANCE:
            continue
        rule_merchant = metadata.get("merchant")
        merchant_rule = isinstance(rule_merchant, str) and bool(rule_merchant.strip())
        if merchant_rule and _normalize_merchant(str(rule_merchant)) != _normalize_merchant(merchant):
            continue
        # A merchant-specific rule wins over a merely similar preference.
        rank = (0 if merchant_rule else 1, distance)
        if best_rank is None or rank < best_rank:
            best_category = category
            best_index = index
            best_rank = rank
    return best_category, best_index


def _rule_payload(result: dict[str, object], index: int, category: str) -> dict[str, object]:
    """JSON-safe description of the Chroma preference row that was applied."""
    metadatas = _first_list(result.get("metadatas"))
    documents = _first_list(result.get("documents"))
    identifiers = _first_list(result.get("ids"))
    distances = _first_list(result.get("distances"))
    metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
    document = documents[index] if index < len(documents) and isinstance(documents[index], str) else None
    rule_id = identifiers[index] if index < len(identifiers) and isinstance(identifiers[index], str) else None
    clean_meta = {
        str(key): value
        for key, value in metadata.items()
        if isinstance(value, str | int | float | bool) or value is None
    }
    merchant = clean_meta.get("merchant")
    return {
        "id": rule_id,
        "document": document,
        "category": category,
        "merchant": merchant if isinstance(merchant, str) else None,
        "distance": _distance_at(distances, index),
        "metadata": clean_meta,
    }


def _category_from_metadata(metadata: dict[object, object], categories_by_slug: dict[str, str]) -> str | None:
    explicit = metadata.get("suggested_category") or metadata.get("category")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    topic = metadata.get("topic")
    if isinstance(topic, str):
        return categories_by_slug.get(topic)
    return None


def _first_list(value: object) -> list[object]:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    if isinstance(first, list):
        return list(first)
    return []


def _normalize_merchant(value: str) -> str:
    collapsed = " ".join(value.casefold().split())
    for mark in ('"', "״", "“", "”", "'", "׳", "\\"):
        collapsed = collapsed.replace(mark, "")
    return " ".join(collapsed.split())


def _is_generic_merchant(value: str) -> bool:
    return _normalize_merchant(value) in _GENERIC_MERCHANT_NAMES


def _distance_at(distances: list[object], index: int) -> float | None:
    if index >= len(distances):
        return None
    value = distances[index]
    if isinstance(value, int | float):
        return float(value)
    return None
