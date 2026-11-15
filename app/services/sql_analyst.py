"""Read-only SQL generation and category spend anomaly detection.

The financial analyst may ask a model for SQLite, but only a single ``SELECT``
(or a ``WITH`` query whose main statement is ``SELECT``) can run. Anomaly
flags are computed in Python from stored transactions, not from model SQL.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Protocol
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.models import Category, Transaction, User
from app.schemas.analysis import GeneratedSQL, SpendAnomaly

logger = logging.getLogger(__name__)

LOCAL_TZ = ZoneInfo("Asia/Jerusalem")
ROLLING_MONTHS = 3
# A month is flagged when spend is strictly greater than this multiple of the
# previous three calendar months. Equal to 30% over the average is not a flag.
AVERAGE_BREACH_MULTIPLE = Decimal("1.30")
MAX_RESULT_ROWS = 200

SCHEMA_DDL = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    telegram_id BIGINT UNIQUE,
    display_name VARCHAR(120) NOT NULL,
    preferred_currency VARCHAR(3) NOT NULL DEFAULT 'ILS',
    created_at DATETIME NOT NULL
);

CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    slug VARCHAR(40) NOT NULL,
    name VARCHAR(80) NOT NULL,
    created_at DATETIME NOT NULL,
    UNIQUE (user_id, name),
    UNIQUE (user_id, slug)
);

CREATE TABLE transactions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    category_id INTEGER NOT NULL REFERENCES categories(id),
    merchant VARCHAR(160) NOT NULL,
    description VARCHAR(255),
    amount NUMERIC(12, 2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'ILS',
    receipt_hash VARCHAR(64),
    transacted_at DATETIME NOT NULL,
    created_at DATETIME NOT NULL
);
""".strip()

_FORBIDDEN_KEYWORDS = frozenset({"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "PRAGMA"})
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[()]")
_USER_ID_EQ = re.compile(r"\buser_id\s*=\s*(\d+)\b", re.IGNORECASE)
_USER_ID_IN = re.compile(r"\buser_id\s+IN\s*\(\s*(\d+)\s*\)", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    """The statement is not a single read-only query for one user."""


class SupportsStructuredOutput(Protocol):
    """Chat model that can return a Pydantic object."""

    def with_structured_output(self, schema: type[BaseModel]) -> object:
        """Return a runnable that parses model output into ``schema``."""


def build_sql_prompt(question: str, user_id: int) -> str:
    """Describe the SQLite schema and require a filter on this user."""
    return (
        "Write one read-only SQLite query that answers the question.\n"
        "Put the statement in the sql field and do not wrap it in Markdown.\n"
        f"The statement MUST restrict rows with user_id = {user_id}. "
        "Never read another user's data.\n"
        "Allowed forms are a single SELECT, or WITH ... SELECT. "
        "Do not use INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, or more than one statement.\n"
        "amount is a positive expense. category.name is the Hebrew label. "
        "transacted_at is stored in UTC. currency is an ISO code such as ILS.\n"
        "Schema:\n"
        f"{SCHEMA_DDL}\n\n"
        f"Question: {question.strip()}"
    )


def generate_sql(
    llm: SupportsStructuredOutput,
    *,
    question: str,
    user_id: int,
) -> GeneratedSQL:
    """Ask the model for one SELECT and strip a Markdown fence if it adds one."""
    if user_id <= 0:
        raise ValueError("user_id must be positive")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question is required")

    try:
        runnable = llm.with_structured_output(GeneratedSQL)
        parsed = getattr(runnable, "invoke")([HumanMessage(content=build_sql_prompt(cleaned, user_id))])
    except Exception:
        logger.exception("Failed to generate SQL for user %s", user_id)
        raise

    generated = parsed if isinstance(parsed, GeneratedSQL) else GeneratedSQL.model_validate(parsed)
    return generated.model_copy(update={"sql": strip_sql_fence(generated.sql)})


def assert_read_only(sql: str) -> str:
    """Return a single SELECT statement, or raise ``UnsafeSQLError``."""
    if "\x00" in sql:
        logger.warning("Rejected SQL that contains a null byte")
        raise UnsafeSQLError("SQL contains a null byte")

    masked = _mask_literals_and_comments(sql)
    statements = [part.strip() for part in masked.split(";") if part.strip()]
    if len(statements) != 1:
        logger.warning("Rejected SQL with %s statements", len(statements))
        raise UnsafeSQLError("Only one SQL statement is allowed")

    tokens = [match.group(0).upper() for match in _TOKEN.finditer(statements[0])]
    if not tokens:
        raise UnsafeSQLError("SQL statement is empty")

    for token in tokens:
        if token in _FORBIDDEN_KEYWORDS:
            logger.warning("Rejected SQL containing %s", token)
            raise UnsafeSQLError(f"{token} is not allowed in read-only queries")

    verb = _main_verb(tokens)
    if verb != "SELECT":
        logger.warning("Rejected SQL whose main statement is %s", verb)
        raise UnsafeSQLError("Only SELECT queries are allowed")

    return _single_statement(sql)


def assert_user_scope(sql: str, user_id: int) -> None:
    """Require the statement to filter with this user's numeric id."""
    if user_id <= 0:
        raise ValueError("user_id must be positive")

    masked = _mask_literals_and_comments(sql)
    found = {int(match) for match in _USER_ID_EQ.findall(masked)}
    found.update(int(match) for match in _USER_ID_IN.findall(masked))
    if found != {user_id}:
        logger.warning("Rejected SQL that is not scoped to user %s", user_id)
        raise UnsafeSQLError(f"Query must filter with user_id = {user_id}")


def execute_readonly_query(session: Session, sql: str) -> list[dict[str, object]]:
    """Run one validated SELECT and return JSON-friendly row dictionaries."""
    statement = assert_read_only(sql)
    connection = session.connection()
    query_only = connection.dialect.name == "sqlite"
    enabled = False
    try:
        if query_only:
            connection.exec_driver_sql("PRAGMA query_only = ON")
            enabled = True
        result = connection.execute(text(statement))
        rows: list[dict[str, object]] = []
        for mapping in result.mappings():
            rows.append({key: _jsonable(value) for key, value in mapping.items()})
            if len(rows) >= MAX_RESULT_ROWS:
                logger.info("Truncating SQL result to %s rows", MAX_RESULT_ROWS)
                break
        return rows
    except Exception:
        logger.exception("Read-only SQL execution failed")
        raise
    finally:
        if enabled:
            try:
                connection.exec_driver_sql("PRAGMA query_only = OFF")
            except Exception:
                logger.exception("Could not restore SQLite query_only")


def detect_spend_anomalies(
    session: Session,
    user_id: int,
    *,
    as_of: date | None = None,
    budget_limits: Mapping[str, Decimal] | None = None,
) -> list[SpendAnomaly]:
    """Flag category months that beat the prior 3-month average or a budget.

    Each calendar month through ``as_of`` (today in Asia/Jerusalem when omitted)
    is the "current" month. Its spend is compared with the average of the three
    preceding calendar months, counting a missing month as zero. A category is
    flagged when that spend is more than 30% above the average, or when it is
    above ``monthly_budget_limit`` for the category slug.
    """
    if session.get(User, user_id) is None:
        raise LookupError(f"User {user_id} does not exist")

    cutoff = as_of if as_of is not None else datetime.now(LOCAL_TZ).date()
    limits = {slug: _money(Decimal(str(limit))) for slug, limit in (budget_limits or {}).items()}

    try:
        loaded = session.execute(
            select(
                Transaction.transacted_at,
                Transaction.amount,
                Transaction.merchant,
                Category.slug,
                Category.name,
            )
            .join(Category, Transaction.category_id == Category.id)
            .where(Transaction.user_id == user_id)
        ).all()
    except Exception:
        logger.exception("Failed to load transactions for user %s", user_id)
        raise

    totals: dict[tuple[int, int, str], Decimal] = defaultdict(lambda: Decimal("0"))
    largest: dict[tuple[int, int, str], tuple[Decimal, str]] = {}
    names: dict[str, str] = {}
    months: set[tuple[int, int]] = set()

    for transacted_at, amount, merchant, slug, name in loaded:
        year, month = _year_month(transacted_at)
        key = (year, month, slug)
        value = Decimal(str(amount))
        totals[key] += value
        names[slug] = name
        months.add((year, month))
        current = largest.get(key)
        if current is None or value > current[0]:
            largest[key] = (value, merchant)

    flags: list[SpendAnomaly] = []
    for year, month in sorted(months):
        if (year, month) > (cutoff.year, cutoff.month):
            continue
        for slug in sorted(names):
            current_spend = totals.get((year, month, slug), Decimal("0"))
            if current_spend == 0:
                continue
            prior = [
                totals.get((prior_year, prior_month, slug), Decimal("0"))
                for prior_year, prior_month in _previous_months(year, month, ROLLING_MONTHS)
            ]
            average = sum(prior, Decimal("0")) / Decimal(ROLLING_MONTHS)
            limit = limits.get(slug)
            exceeded_average = average > 0 and current_spend > average * AVERAGE_BREACH_MULTIPLE
            exceeded_budget = limit is not None and current_spend > limit
            if not exceeded_average and not exceeded_budget:
                continue
            top_amount, top_merchant = largest[(year, month, slug)]
            flags.append(
                SpendAnomaly(
                    category_slug=slug,
                    category_name=names[slug],
                    year=year,
                    month=month,
                    current_spend=_money(current_spend),
                    rolling_average=_money(average),
                    monthly_budget_limit=limit,
                    exceeded_average=exceeded_average,
                    exceeded_budget=exceeded_budget,
                    largest_merchant=top_merchant,
                    largest_amount=_money(top_amount),
                )
            )
    return flags


def strip_sql_fence(sql: str) -> str:
    """Remove a single surrounding Markdown code fence."""
    text = sql.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _single_statement(sql: str) -> str:
    """Return the original statement without a trailing semicolon."""
    return sql[:_statement_end(sql)].strip()


def _statement_end(sql: str) -> int:
    """Index of the first semicolon outside comments and quotes, or ``len(sql)``."""
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        nxt = sql[index + 1] if index + 1 < length else ""
        if char == "-" and nxt == "-":
            index += 2
            while index < length and sql[index] not in "\n\r":
                index += 1
            continue
        if char == "/" and nxt == "*":
            index += 2
            while index + 1 < length and not (sql[index] == "*" and sql[index + 1] == "/"):
                index += 1
            index = min(length, index + 2)
            continue
        if char == "'":
            index = _skip_quoted(sql, index)
            continue
        if char == ";":
            return index
        index += 1
    return length


def _main_verb(tokens: list[str]) -> str:
    if tokens[0] != "WITH":
        return tokens[0]

    depth = 0
    for token in tokens[1:]:
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and token == "SELECT":
            return "SELECT"
        elif depth == 0 and token in _FORBIDDEN_KEYWORDS:
            return token
    raise UnsafeSQLError("WITH queries must end in SELECT")


def _mask_literals_and_comments(sql: str) -> str:
    """Replace comments and quoted strings so keywords inside them are ignored."""
    out: list[str] = []
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        nxt = sql[index + 1] if index + 1 < length else ""
        if char == "-" and nxt == "-":
            index += 2
            while index < length and sql[index] not in "\n\r":
                index += 1
            out.append(" ")
            continue
        if char == "/" and nxt == "*":
            index += 2
            while index + 1 < length and not (sql[index] == "*" and sql[index + 1] == "/"):
                index += 1
            index = min(length, index + 2)
            out.append(" ")
            continue
        if char == "'":
            end = _skip_quoted(sql, index)
            out.append("''")
            index = end
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _skip_quoted(sql: str, index: int) -> int:
    """Return the index just after a single-quoted literal that starts at ``index``."""
    index += 1
    length = len(sql)
    while index < length:
        if sql[index] == "'":
            if index + 1 < length and sql[index + 1] == "'":
                index += 2
                continue
            return index + 1
        index += 1
    return length


def _previous_months(year: int, month: int, count: int) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    cursor_year, cursor_month = year, month
    for _ in range(count):
        cursor_month -= 1
        if cursor_month == 0:
            cursor_month = 12
            cursor_year -= 1
        found.append((cursor_year, cursor_month))
    return found


def _year_month(value: datetime | str) -> tuple[int, int]:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
    else:
        parsed = value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone(LOCAL_TZ)
    return local.year, local.month


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def _jsonable(value: object) -> object:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
