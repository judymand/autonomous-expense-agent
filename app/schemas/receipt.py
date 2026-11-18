"""Structured receipt fields returned by the vision model."""

from datetime import date
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


class ReceiptItem(BaseModel):
    """One line on a receipt."""

    name: str = Field(min_length=1)
    quantity: Decimal | None = None
    amount: Decimal | None = None


class ReceiptExtraction(BaseModel):
    """Normalized purchase extracted from a receipt image or PDF."""

    merchant_name: str = Field(min_length=1)
    transaction_date: date
    total_amount: Annotated[Decimal, Field(ge=0)]
    currency: str = Field(default="ILS", min_length=1, max_length=8)
    items: list[ReceiptItem] = Field(default_factory=list)
    suggested_category: str = Field(min_length=1)
    confidence_score: Annotated[float, Field(ge=0, le=1)]

    @field_validator("items", mode="before")
    @classmethod
    def _coerce_items(cls, value: object) -> object:
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        coerced: list[object] = []
        for item in value:
            if isinstance(item, str):
                coerced.append({"name": item})
            else:
                coerced.append(item)
        return coerced

    @field_validator("currency")
    @classmethod
    def _strip_currency(cls, value: str) -> str:
        return value.strip()


class IngestionResult(BaseModel):
    """Receipt extraction plus duplicate and preference decisions."""

    extraction: ReceiptExtraction
    receipt_hash: str
    is_duplicate: bool
    matched_transaction_id: int | None = None
    preference_applied: bool = False
    matched_rule: dict[str, Any] | None = None
