"""Financial analyst: turn a question into read-only SQL and anomaly flags."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.models import User
from app.schemas.analysis import AnalystReport
from app.services.llm import get_chat_model
from app.services.sql_analyst import (
    SupportsStructuredOutput,
    UnsafeSQLError,
    assert_read_only,
    assert_user_scope,
    detect_spend_anomalies,
    execute_readonly_query,
    generate_sql,
)

logger = logging.getLogger(__name__)


def run_financial_analyst(
    question: str,
    *,
    user_id: int,
    session: Session,
    llm: SupportsStructuredOutput | None = None,
    as_of: date | None = None,
    budget_limits: Mapping[str, Decimal] | None = None,
) -> AnalystReport:
    """Answer ``question`` with scoped SELECT rows and programmatic anomaly flags."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question is required")
    if session.get(User, user_id) is None:
        raise LookupError(f"User {user_id} does not exist")

    model = llm if llm is not None else get_chat_model()
    try:
        generated = generate_sql(model, question=cleaned, user_id=user_id)
        statement = assert_read_only(generated.sql)
        assert_user_scope(statement, user_id)
        rows = execute_readonly_query(session, statement)
        anomalies = detect_spend_anomalies(
            session,
            user_id,
            as_of=as_of,
            budget_limits=budget_limits,
        )
    except UnsafeSQLError:
        raise
    except Exception:
        logger.exception("Financial analyst failed for user %s", user_id)
        raise

    return AnalystReport(
        user_id=user_id,
        question=cleaned,
        sql=statement,
        rows=rows,
        anomalies=anomalies,
    )
