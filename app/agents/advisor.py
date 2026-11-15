"""Advisor: turn analyst rows and anomaly flags into concise Hebrew guidance."""

from __future__ import annotations

import json
import logging
from typing import Protocol

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.schemas.analysis import AdvisorDraft, AdvisorResponse, AnalystReport, SpendAnomaly
from app.services.llm import get_chat_model

logger = logging.getLogger(__name__)


class SupportsStructuredOutput(Protocol):
    """Chat model that can return a Pydantic object."""

    def with_structured_output(self, schema: type[BaseModel]) -> object:
        """Return a runnable that parses model output into ``schema``."""


def run_advisor(
    report: AnalystReport,
    *,
    llm: SupportsStructuredOutput | None = None,
) -> AdvisorResponse:
    """Synthesize an analyst report into a short Hebrew message."""
    model = llm if llm is not None else get_chat_model()
    draft = request_advice(model, report)
    message = format_advisor_message(draft, report.anomalies)
    return AdvisorResponse(
        message_he=message,
        summary=draft.summary.strip(),
        recommendations=[item.strip() for item in draft.recommendations if item.strip()],
    )


def request_advice(llm: SupportsStructuredOutput, report: AnalystReport) -> AdvisorDraft:
    """Ask the model for a Hebrew summary and a few practical recommendations."""
    try:
        runnable = llm.with_structured_output(AdvisorDraft)
        parsed = getattr(runnable, "invoke")([HumanMessage(content=build_advisor_prompt(report))])
    except Exception:
        logger.exception("Advisor model failed for user %s", report.user_id)
        raise
    if isinstance(parsed, AdvisorDraft):
        return parsed
    return AdvisorDraft.model_validate(parsed)


def format_advisor_message(draft: AdvisorDraft, anomalies: list[SpendAnomaly] | None = None) -> str:
    """Combine the model narrative with the detected anomaly figures."""
    summary = draft.summary.strip()
    if not summary:
        raise ValueError("Advisor summary is empty")

    lines = [summary]
    flags = list(anomalies or [])
    if flags:
        lines.append("")
        lines.append("חריגות שזוהו:")
        lines.extend(f"• {_anomaly_line(anomaly)}" for anomaly in flags)

    actions = [item.strip() for item in draft.recommendations if item.strip()]
    if actions:
        lines.append("")
        lines.append("מה כדאי לעשות:")
        lines.extend(f"• {action}" for action in actions)
    return "\n".join(lines)


def build_advisor_prompt(report: AnalystReport) -> str:
    """Give the model the question, rows, and anomaly flags to explain in Hebrew."""
    payload = {
        "question": report.question,
        "sql": report.sql,
        "rows": report.rows[:30],
        "anomalies": [anomaly.model_dump(mode="json") for anomaly in report.anomalies],
    }
    return (
        "You are a personal finance advisor. Write the summary and recommendations "
        "in natural, concise Hebrew. Use only numbers from the report. Explain the "
        "meaningful anomalies and suggest practical next steps. Do not mention SQL, "
        "table names, or column names.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _anomaly_line(anomaly: SpendAnomaly) -> str:
    period = f"{anomaly.month:02d}/{anomaly.year}"
    parts = [f"{anomaly.category_name} ב-{period}: {anomaly.current_spend:.2f} ש״ח"]
    if anomaly.exceeded_average:
        parts.append(f"מעל הממוצע של {anomaly.rolling_average:.2f} ש״ח")
    if anomaly.exceeded_budget and anomaly.monthly_budget_limit is not None:
        parts.append(f"מעל התקציב החודשי של {anomaly.monthly_budget_limit:.2f} ש״ח")
    if anomaly.largest_merchant:
        parts.append(f"העסקה הגדולה אצל {anomaly.largest_merchant} ({anomaly.largest_amount:.2f} ש״ח)")
    return " — ".join(parts)
