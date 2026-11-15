"""Guardrails, seeded anomaly detection, and advisor formatting."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.agents.advisor import format_advisor_message, run_advisor
from app.agents.analyst import run_financial_analyst
from app.db.models import Base, Category, Transaction, User
from app.db.seed import ANOMALIES, DEMO_TELEGRAM_ID
from app.db.settings import PROJECT_ROOT
from app.schemas.analysis import AdvisorDraft, AnalystReport, GeneratedSQL, SpendAnomaly
from app.services.sql_analyst import (
    SCHEMA_DDL,
    UnsafeSQLError,
    assert_read_only,
    assert_user_scope,
    detect_spend_anomalies,
    execute_readonly_query,
    generate_sql,
)

SEEDED_DB = PROJECT_ROOT / "data" / "expenses.sqlite3"


class _Runnable:
    def __init__(self, parent: FakeLLM, schema: type[object]) -> None:
        self._parent = parent
        self._schema = schema

    def invoke(self, messages: list[object]) -> object:
        content = getattr(messages[0], "content", messages[0])
        self._parent.prompts.append(content if isinstance(content, str) else str(content))
        payload = self._parent.payloads[self._schema]
        if isinstance(payload, Exception):
            raise payload
        return payload


class FakeLLM:
    """Returns canned structured output and records the prompt."""

    def __init__(self, payloads: dict[type[object], object]) -> None:
        self.payloads = payloads
        self.prompts: list[str] = []

    def with_structured_output(self, schema: type[object]) -> _Runnable:
        return _Runnable(self, schema)


def _session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()


def _user(session: Session) -> User:
    user = User(telegram_id=77, display_name="בדיקה", preferred_currency="ILS")
    session.add(user)
    session.flush()
    groceries = Category(user_id=user.id, slug="groceries", name="סופר ומכולת")
    shopping = Category(user_id=user.id, slug="shopping", name="קניות")
    session.add_all([groceries, shopping])
    session.flush()
    return user


def _add(
    session: Session,
    user: User,
    slug: str,
    when: datetime,
    amount: str,
    merchant: str,
) -> None:
    category_id = session.scalar(select(Category.id).where(Category.user_id == user.id, Category.slug == slug))
    assert category_id is not None
    session.add(
        Transaction(
            user_id=user.id,
            category_id=category_id,
            merchant=merchant,
            amount=Decimal(amount),
            currency="ILS",
            transacted_at=when,
        )
    )
    session.flush()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT amount FROM transactions WHERE user_id = 1",
        "SELECT amount FROM transactions WHERE user_id = 1;",
        "WITH totals AS (SELECT SUM(amount) AS total FROM transactions WHERE user_id = 1) "
        "SELECT total FROM totals",
        "SELECT '; DROP TABLE transactions' AS note FROM transactions WHERE user_id = 1",
        "SELECT * FROM transactions WHERE description = 'DELETE' AND user_id = 1",
        "SELECT 1 -- ; DROP TABLE transactions",
        "SELECT merchant FROM transactions WHERE merchant = 'PRAGMA' AND user_id = 1",
    ],
)
def test_read_only_guard_allows_select(sql: str) -> None:
    assert assert_read_only(sql).lower().lstrip().startswith(("select", "with"))


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE transactions",
        "DELETE FROM transactions",
        "dElEtE FROM transactions WHERE user_id = 1",
        "UPDATE transactions SET amount = 0",
        "INSERT INTO transactions (user_id, amount) VALUES (1, 1)",
        "ALTER TABLE transactions ADD COLUMN hacked TEXT",
        "PRAGMA table_info(transactions)",
        "SELECT * FROM transactions; DROP TABLE transactions",
        "SELECT 1; SELECT 2",
        "WITH x AS (SELECT 1) DELETE FROM transactions",
        "/* comment */ DROP TABLE users",
        "SELECT * FROM transactions WHERE user_id = 1; UPDATE transactions SET amount = 0",
        "SELECT 1\x00; DROP TABLE transactions",
    ],
)
def test_read_only_guard_rejects_mutations(sql: str) -> None:
    with pytest.raises(UnsafeSQLError):
        assert_read_only(sql)


def test_user_scope_requires_the_requested_id() -> None:
    assert_user_scope("SELECT * FROM transactions WHERE user_id = 4", 4)
    with pytest.raises(UnsafeSQLError):
        assert_user_scope("SELECT * FROM transactions", 4)
    with pytest.raises(UnsafeSQLError):
        assert_user_scope("SELECT * FROM transactions WHERE user_id = 4 OR user_id = 5", 4)
    with pytest.raises(UnsafeSQLError):
        assert_user_scope("SELECT 'user_id = 4' FROM transactions", 4)


def test_execute_returns_rows_and_does_not_apply_rejected_sql() -> None:
    session = _session()
    user = _user(session)
    _add(session, user, "groceries", datetime(2026, 1, 15, tzinfo=timezone.utc), "40.00", "שופרסל")

    rows = execute_readonly_query(
        session,
        f"SELECT merchant, amount FROM transactions WHERE user_id = {user.id}",
    )
    assert rows[0]["merchant"] == "שופרסל"
    assert Decimal(str(rows[0]["amount"])) == Decimal("40.00")

    with pytest.raises(UnsafeSQLError):
        execute_readonly_query(session, "DELETE FROM transactions")
    remaining = session.scalar(select(func.count()).select_from(Transaction))
    assert remaining == 1
    session.close()


def test_schema_prompt_includes_tables_and_user_filter() -> None:
    llm = FakeLLM({GeneratedSQL: GeneratedSQL(sql="SELECT 1")})
    generated = generate_sql(llm, question="כמה הוצאתי על סופר?", user_id=7)
    assert generated.sql == "SELECT 1"
    prompt = llm.prompts[0]
    assert "CREATE TABLE users" in prompt
    assert "CREATE TABLE categories" in prompt
    assert "CREATE TABLE transactions" in prompt
    assert "receipt_hash" in SCHEMA_DDL
    assert "user_id = 7" in prompt
    assert "כמה הוצאתי על סופר?" in prompt


def test_anomaly_thresholds_for_average_and_budget() -> None:
    session = _session()
    user = _user(session)
    for month, grocery, shopping in (
        (1, "100.00", "10.00"),
        (2, "100.00", "10.00"),
        (3, "100.00", "10.00"),
        (4, "130.00", "40.00"),
        (5, "9000.00", "10.00"),
    ):
        when = datetime(2026, month, 15, 12, tzinfo=timezone.utc)
        _add(session, user, "groceries", when, grocery, "שופרסל")
        _add(session, user, "shopping", when, shopping, "KSP")

    flags = detect_spend_anomalies(
        session,
        user.id,
        as_of=date(2026, 4, 30),
        budget_limits={"groceries": Decimal("100.00"), "shopping": Decimal("500.00")},
    )
    by_slug = {flag.category_slug: flag for flag in flags if flag.month == 4}
    assert set(by_slug) == {"groceries", "shopping"}

    groceries = by_slug["groceries"]
    assert groceries.exceeded_budget is True
    assert groceries.exceeded_average is False
    assert groceries.monthly_budget_limit == Decimal("100.00")
    assert groceries.current_spend == Decimal("130.00")

    shopping = by_slug["shopping"]
    assert shopping.exceeded_average is True
    assert shopping.exceeded_budget is False
    assert shopping.rolling_average == Decimal("10.00")

    assert all(flag.month != 5 for flag in flags)
    session.close()


def test_seeded_database_flags_planted_outliers() -> None:
    assert SEEDED_DB.is_file()
    engine = create_engine(f"sqlite:///{SEEDED_DB}")
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with factory() as session:
        user = session.scalar(select(User).where(User.telegram_id == DEMO_TELEGRAM_ID))
        assert user is not None
        flags = detect_spend_anomalies(session, user.id, as_of=date(2026, 11, 15))

    found = {
        (flag.year, flag.month, flag.category_slug, flag.largest_merchant, flag.largest_amount)
        for flag in flags
    }
    for planted_on, slug, merchant, _description, amount in ANOMALIES:
        assert (planted_on.year, planted_on.month, slug, merchant, Decimal(amount)) in found
        match = next(
            flag
            for flag in flags
            if flag.year == planted_on.year and flag.month == planted_on.month and flag.category_slug == slug
        )
        assert match.exceeded_average is True
        assert match.current_spend > match.rolling_average * Decimal("1.30")


def test_analyst_rejects_model_sql_without_changing_rows() -> None:
    session = _session()
    user = _user(session)
    _add(session, user, "groceries", datetime(2026, 1, 15, tzinfo=timezone.utc), "40.00", "שופרסל")
    llm = FakeLLM({GeneratedSQL: GeneratedSQL(sql="DELETE FROM transactions")})

    with pytest.raises(UnsafeSQLError):
        run_financial_analyst("תמחק הכל", user_id=user.id, session=session, llm=llm)

    assert session.scalar(select(func.count()).select_from(Transaction)) == 1
    session.close()


def test_analyst_and_advisor_with_mocked_models() -> None:
    session = _session()
    user = _user(session)
    for month in (1, 2, 3):
        when = datetime(2026, month, 10, 9, tzinfo=timezone.utc)
        _add(session, user, "groceries", when, "80.00", "רמי לוי")
    _add(
        session,
        user,
        "groceries",
        datetime(2026, 4, 10, 9, tzinfo=timezone.utc),
        "150.00",
        "שופרסל",
    )

    analyst = FakeLLM(
        {
            GeneratedSQL: GeneratedSQL(
                sql=f"SELECT merchant, amount FROM transactions WHERE user_id = {user.id} ORDER BY amount DESC"
            )
        }
    )
    report = run_financial_analyst(
        "מה חריג באפריל?",
        user_id=user.id,
        session=session,
        llm=analyst,
        as_of=date(2026, 4, 30),
        budget_limits={"groceries": Decimal("100.00")},
    )
    assert report.rows[0]["merchant"] == "שופרסל"
    assert "CREATE TABLE transactions" in analyst.prompts[0]
    grocery_flags = [flag for flag in report.anomalies if flag.category_slug == "groceries" and flag.month == 4]
    assert len(grocery_flags) == 1
    assert grocery_flags[0].exceeded_average is True
    assert grocery_flags[0].exceeded_budget is True

    draft = AdvisorDraft(
        summary="באפריל הסופר חרג גם מהממוצע וגם מהתקציב.",
        recommendations=["הגדירו תקרת סופר של 100 ש״ח", "דחו קניות גדולות"],
    )
    advisor = FakeLLM({AdvisorDraft: draft})
    response = run_advisor(report, llm=advisor)

    assert response.summary == draft.summary
    assert response.message_he.startswith(draft.summary)
    assert "• הגדירו תקרת סופר של 100 ש״ח" in response.message_he
    assert "• דחו קניות גדולות" in response.message_he
    assert "סופר ומכולת" in response.message_he
    assert "שופרסל" in response.message_he
    assert "סופר ומכולת" in advisor.prompts[0]
    assert "150.00" in advisor.prompts[0]
    session.close()


def test_advisor_formatting_without_anomalies() -> None:
    draft = AdvisorDraft(summary="אין חריגות בולטות החודש.", recommendations=[])
    assert format_advisor_message(draft, []) == draft.summary

    report = AnalystReport(
        user_id=1,
        question="סיכום",
        sql="SELECT 1",
        rows=[],
        anomalies=[
            SpendAnomaly(
                category_slug="shopping",
                category_name="קניות",
                year=2026,
                month=10,
                current_spend=Decimal("9665.51"),
                rolling_average=Decimal("68.65"),
                monthly_budget_limit=None,
                exceeded_average=True,
                exceeded_budget=False,
                largest_merchant="KSP",
                largest_amount=Decimal("8990.00"),
            )
        ],
    )
    rendered = format_advisor_message(
        AdvisorDraft(summary="רכישת המחשב חריגה.", recommendations=["פרסו את הקניות הגדולות"]),
        report.anomalies,
    )
    assert "קניות ב-10/2026" in rendered
    assert "KSP" in rendered
    assert "מה כדאי לעשות:" in rendered
