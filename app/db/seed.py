"""Seed a demo user, Hebrew categories, and six months of ILS expenses.

Running this module replaces previously seeded relational rows and rebuilds the
``user_preferences`` collection. Three purchases are intentionally far outside
their category's normal range so later anomaly checks have a known target.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from faker import Faker
from sqlalchemy import delete, func, select

from app.db.models import Category, Transaction, User
from app.db.receipt_hash import compute_receipt_hash
from app.db.session import get_session, init_db
from app.db.settings import get_settings
from app.db.vector_store import (
    USER_PREFERENCES_COLLECTION,
    get_chroma_client,
    get_user_preferences_collection,
)

logger = logging.getLogger(__name__)

DEMO_TELEGRAM_ID = 100_000_001
HISTORY_START = date(2026, 5, 16)
HISTORY_END = date(2026, 11, 15)
LOCAL_TZ = ZoneInfo("Asia/Jerusalem")
RNG_SEED = 42

CATEGORIES: tuple[tuple[str, str], ...] = (
    ("groceries", "סופר ומכולת"),
    ("restaurants", "מסעדות ובתי קפה"),
    ("transport", "תחבורה"),
    ("housing", "דיור וחשבונות"),
    ("health", "בריאות"),
    ("leisure", "בילויים"),
    ("shopping", "קניות"),
    ("telecom", "תקשורת"),
)

GROCERY_MERCHANTS = ("שופרסל", "רמי לוי", "ויקטורי", "אושר עד", "יוחננוף", "טיב טעם")
RESTAURANT_MERCHANTS = (
    "ארומה",
    "קפה קפה",
    "לנדוור",
    "מקדונלד'ס",
    "פלאפל הכרמל",
    "שווארמה הגליל",
)
TRANSPORT_MERCHANTS = ("רכבת ישראל", "רב-קו", "פז", "דלק", "גט", "יאנגו")
HEALTH_MERCHANTS = ("סופר-פארם", "כללית", "מכבי", "בי-גוד")
LEISURE_MERCHANTS = ("סינמה סיטי", "הוט סינמה", "כרטיסים להופעה")
SHOPPING_MERCHANTS = ("קסטרו", "זארה", "H&M", "איקאה", "פוקס")

# Clear outliers against the everyday ranges used by the generator above.
ANOMALIES: tuple[tuple[date, str, str, str, str], ...] = (
    (date(2026, 7, 18), "groceries", "שופרסל", "סל גדול לפני החגים", "4750.40"),
    (date(2026, 9, 5), "restaurants", "מסעדת מחניודה", "ארוחת ערב", "3480.00"),
    (date(2026, 10, 21), "shopping", "KSP", "מחשב נייד", "8990.00"),
)

PREFERENCES: tuple[tuple[str, str, str], ...] = (
    (
        "pref-groceries",
        "groceries",
        "המשתמש מעדיף סופרים זולים כמו רמי לוי ואושר עד, וקונה מכולת בדרך כלל ביום ראשון או חמישי.",
    ),
    (
        "pref-restaurants",
        "restaurants",
        "תקציב המסעדות החודשי הרגיל הוא בערך 800 ש״ח. ארוחות יקרות במיוחד אינן שגרתיות.",
    ),
    (
        "pref-telecom",
        "telecom",
        "חשבון התקשורת הקבוע נע סביב 90 ש״ח לחודש, כולל סלולר ואינטרנט.",
    ),
)


@dataclass(frozen=True)
class CategorySpend:
    """Aggregated spend for one Hebrew category."""

    name: str
    transaction_count: int
    total_ils: Decimal


@dataclass(frozen=True)
class PlantedAnomaly:
    """One seeded purchase that should stand out in later analysis."""

    transacted_on: date
    category_name: str
    merchant: str
    amount_ils: Decimal


@dataclass(frozen=True)
class SeedSummary:
    """Counts and totals read back after seeding."""

    user_id: int
    display_name: str
    telegram_id: int
    categories: tuple[str, ...]
    transaction_count: int
    first_transaction_on: date
    last_transaction_on: date
    total_ils: Decimal
    spend_by_category: tuple[CategorySpend, ...]
    anomalies: tuple[PlantedAnomaly, ...]
    preference_count: int
    chroma_path: str


def _money(rng: random.Random, low: float, high: float) -> Decimal:
    cents = rng.randint(int(low * 100), int(high * 100))
    return (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))


def _at(day: date, hour: int, minute: int) -> datetime:
    local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=LOCAL_TZ)
    return local.astimezone(timezone.utc)


def _random_at(rng: random.Random, day: date) -> datetime:
    return _at(day, rng.randint(8, 21), rng.randint(0, 59))


def _daterange(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _month_starts(start: date, end: date) -> list[date]:
    cursor = date(start.year, start.month, 1)
    if cursor < start:
        month = cursor.month + 1
        year = cursor.year + (1 if month == 13 else 0)
        month = 1 if month == 13 else month
        cursor = date(year, month, 1)
    found: list[date] = []
    while cursor <= end:
        found.append(cursor)
        month = cursor.month + 1
        year = cursor.year + (1 if month == 13 else 0)
        month = 1 if month == 13 else month
        cursor = date(year, month, 1)
    return found


def _build_transactions(
    rng: random.Random,
    user_id: int,
    category_ids: dict[str, int],
) -> list[Transaction]:
    rows: list[Transaction] = []

    def add(
        day: date,
        slug: str,
        merchant: str,
        amount: Decimal,
        description: str | None = None,
        hour: int | None = None,
        minute: int | None = None,
    ) -> None:
        when = (
            _at(day, hour, minute)
            if hour is not None and minute is not None
            else _random_at(rng, day)
        )
        rows.append(
            Transaction(
                user_id=user_id,
                category_id=category_ids[slug],
                merchant=merchant,
                description=description,
                amount=amount,
                currency="ILS",
                receipt_hash=compute_receipt_hash(merchant, day, amount, "ILS"),
                transacted_at=when,
            )
        )

    for day in _daterange(HISTORY_START, HISTORY_END):
        weekday = day.weekday()
        if weekday in (6, 3) or (weekday == 1 and rng.random() < 0.35):
            add(day, "groceries", rng.choice(GROCERY_MERCHANTS), _money(rng, 85, 420))
        if rng.random() < (0.55 if weekday in (4, 5) else 0.32):
            add(day, "restaurants", rng.choice(RESTAURANT_MERCHANTS), _money(rng, 28, 145))
        if rng.random() < 0.28:
            add(day, "transport", rng.choice(TRANSPORT_MERCHANTS), _money(rng, 12, 260))
        if rng.random() < 0.08:
            add(day, "health", rng.choice(HEALTH_MERCHANTS), _money(rng, 25, 180))
        if rng.random() < 0.07:
            add(day, "leisure", rng.choice(LEISURE_MERCHANTS), _money(rng, 39, 220))
        if rng.random() < 0.06:
            add(day, "shopping", rng.choice(SHOPPING_MERCHANTS), _money(rng, 59, 390))

    for month_start in _month_starts(HISTORY_START, HISTORY_END):
        rent_day = month_start
        utilities_day = min(date(month_start.year, month_start.month, 10), HISTORY_END)
        telecom_day = min(date(month_start.year, month_start.month, 5), HISTORY_END)
        streaming_day = min(date(month_start.year, month_start.month, 8), HISTORY_END)
        add(rent_day, "housing", "שכר דירה", Decimal("6200.00"), "שכר דירה חודשי", 8, 5)
        add(
            utilities_day,
            "housing",
            "חברת החשמל",
            _money(rng, 180, 420),
            "חשבון חשמל",
            9,
            15,
        )
        add(utilities_day, "housing", "עירייה", Decimal("385.00"), "ארנונה", 9, 20)
        add(utilities_day, "housing", "תאגיד המים", _money(rng, 55, 130), "מים וביוב", 9, 25)
        add(telecom_day, "telecom", rng.choice(("סלקום", "פרטנר", "בזק")), _money(rng, 79, 119), "חשבון תקשורת", 10, 0)
        add(streaming_day, "leisure", "נטפליקס", Decimal("49.90"), "מנוי חודשי", 11, 0)
        add(streaming_day, "leisure", "ספוטיפיי", Decimal("21.90"), "מנוי חודשי", 11, 5)

    for day, slug, merchant, description, amount in ANOMALIES:
        add(day, slug, merchant, Decimal(amount), description, 19, 30)

    return rows


def _reset_relational_data() -> None:
    with get_session() as session:
        session.execute(delete(Transaction))
        session.execute(delete(Category))
        session.execute(delete(User))


def _seed_preferences(telegram_id: int) -> int:
    client = get_chroma_client()
    names: set[str] = set()
    for collection in client.list_collections():
        name = collection if isinstance(collection, str) else getattr(collection, "name", None)
        if isinstance(name, str):
            names.add(name)
    if USER_PREFERENCES_COLLECTION in names:
        client.delete_collection(USER_PREFERENCES_COLLECTION)

    collection = get_user_preferences_collection(client)
    collection.add(
        ids=[item_id for item_id, _topic, _document in PREFERENCES],
        documents=[document for _item_id, _topic, document in PREFERENCES],
        metadatas=[
            {"telegram_id": telegram_id, "topic": topic, "language": "he"}
            for _item_id, topic, _document in PREFERENCES
        ],
    )
    return int(collection.count())


def _read_summary(preference_count: int) -> SeedSummary:
    with get_session() as session:
        user = session.scalar(select(User).where(User.telegram_id == DEMO_TELEGRAM_ID))
        if user is None or user.telegram_id is None:
            raise RuntimeError("Demo user was not persisted")

        category_names = tuple(
            session.scalars(select(Category.name).where(Category.user_id == user.id).order_by(Category.name)).all()
        )
        transaction_count = session.scalar(
            select(func.count(Transaction.id)).where(Transaction.user_id == user.id)
        )
        first_at = session.scalar(select(func.min(Transaction.transacted_at)).where(Transaction.user_id == user.id))
        last_at = session.scalar(select(func.max(Transaction.transacted_at)).where(Transaction.user_id == user.id))
        total = session.scalar(select(func.sum(Transaction.amount)).where(Transaction.user_id == user.id))
        grouped = session.execute(
            select(Category.name, func.count(Transaction.id), func.sum(Transaction.amount))
            .join(Transaction, Transaction.category_id == Category.id)
            .where(Transaction.user_id == user.id)
            .group_by(Category.name)
            .order_by(Category.name)
        ).all()

        if transaction_count is None or first_at is None or last_at is None or total is None:
            raise RuntimeError("Seeded transactions could not be read back")

        anomaly_rows: list[PlantedAnomaly] = []
        for day, _slug, merchant, _description, amount in ANOMALIES:
            match = session.scalar(
                select(Transaction)
                .join(Category)
                .where(
                    Transaction.user_id == user.id,
                    Transaction.merchant == merchant,
                    Transaction.amount == Decimal(amount),
                )
            )
            if match is None:
                raise RuntimeError(f"Planted anomaly for {merchant} was not stored")
            anomaly_rows.append(
                PlantedAnomaly(
                    transacted_on=day,
                    category_name=match.category.name,
                    merchant=merchant,
                    amount_ils=Decimal(amount),
                )
            )

    spend = tuple(
        CategorySpend(name=name, transaction_count=int(count), total_ils=Decimal(total_amount))
        for name, count, total_amount in grouped
    )
    return SeedSummary(
        user_id=user.id,
        display_name=user.display_name,
        telegram_id=user.telegram_id,
        categories=category_names,
        transaction_count=int(transaction_count),
        first_transaction_on=_as_local_date(first_at),
        last_transaction_on=_as_local_date(last_at),
        total_ils=Decimal(total).quantize(Decimal("0.01")),
        spend_by_category=spend,
        anomalies=tuple(anomaly_rows),
        preference_count=preference_count,
        chroma_path=str(get_settings().resolved_chroma_dir()),
    )


def _as_local_date(value: datetime) -> date:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(LOCAL_TZ).date()


def seed_demo_data() -> SeedSummary:
    """Replace local demo data and return the totals read back from storage."""
    faker = Faker("he_IL")
    faker.seed_instance(RNG_SEED)
    rng = random.Random(RNG_SEED)

    init_db()
    _reset_relational_data()

    with get_session() as session:
        user = User(
            telegram_id=DEMO_TELEGRAM_ID,
            display_name=faker.name(),
            preferred_currency="ILS",
        )
        session.add(user)
        session.flush()

        categories = [
            Category(user_id=user.id, slug=slug, name=name) for slug, name in CATEGORIES
        ]
        session.add_all(categories)
        session.flush()
        category_ids = {category.slug: category.id for category in categories}
        session.add_all(_build_transactions(rng, user.id, category_ids))

    preference_count = _seed_preferences(DEMO_TELEGRAM_ID)
    return _read_summary(preference_count)


def format_summary(summary: SeedSummary) -> str:
    """Render a multi-line English summary that keeps Hebrew names intact."""
    lines = [
        f"User #{summary.user_id}: {summary.display_name} (telegram_id={summary.telegram_id})",
        f"Categories ({len(summary.categories)}): {', '.join(summary.categories)}",
        (
            f"Transactions: {summary.transaction_count} "
            f"from {summary.first_transaction_on.isoformat()} "
            f"to {summary.last_transaction_on.isoformat()}"
        ),
        f"Total spend: {summary.total_ils:,.2f} ILS",
        "Spend by category:",
    ]
    lines.extend(
        f"  {row.name}: {row.transaction_count} txns, {row.total_ils:,.2f} ILS"
        for row in summary.spend_by_category
    )
    lines.append("Planted anomalies:")
    lines.extend(
        f"  {row.transacted_on.isoformat()} {row.category_name} / {row.merchant}: {row.amount_ils:,.2f} ILS"
        for row in summary.anomalies
    )
    lines.append(
        f"Chroma collection '{USER_PREFERENCES_COLLECTION}': "
        f"{summary.preference_count} documents at {summary.chroma_path}"
    )
    return "\n".join(lines)


def main() -> None:
    """Seed local storage and log the resulting summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        summary = seed_demo_data()
    except Exception:
        logger.exception("Failed to seed local demo data")
        raise
    logger.info("Seed completed\n%s", format_summary(summary))


if __name__ == "__main__":
    main()
