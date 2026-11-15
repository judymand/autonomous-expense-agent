"""Stable identity for a receipt: merchant, calendar date, amount, and currency."""

from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha256
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Asia/Jerusalem")


def receipt_calendar_date(value: date | datetime) -> date:
    """Return the purchase date in Asia/Jerusalem when ``value`` is a timestamp."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(LOCAL_TZ).date()
    return value


def compute_receipt_hash(
    merchant: str,
    transacted_on: date | datetime,
    amount: Decimal | str | float,
    currency: str = "ILS",
) -> str:
    """Hash the fields that identify the same purchase across receipts and history."""
    merchant_key = " ".join(merchant.casefold().split())
    day = receipt_calendar_date(transacted_on).isoformat()
    quantized = _as_decimal(amount).quantize(Decimal("0.01"))
    currency_key = currency.strip().upper()
    payload = f"{merchant_key}|{day}|{quantized}|{currency_key}"
    return sha256(payload.encode("utf-8")).hexdigest()


def _as_decimal(amount: Decimal | str | float) -> Decimal:
    if isinstance(amount, Decimal):
        return amount
    return Decimal(str(amount))
