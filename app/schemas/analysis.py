"""Structured outputs for the financial analyst and advisor."""

from decimal import Decimal

from pydantic import BaseModel, Field


class GeneratedSQL(BaseModel):
    """One SQLite statement produced for the financial analyst."""

    sql: str = Field(min_length=1)


class SpendAnomaly(BaseModel):
    """A category whose spend in one month broke a baseline or a budget."""

    category_slug: str
    category_name: str
    year: int = Field(ge=1)
    month: int = Field(ge=1, le=12)
    current_spend: Decimal
    rolling_average: Decimal
    monthly_budget_limit: Decimal | None = None
    exceeded_average: bool
    exceeded_budget: bool
    largest_merchant: str
    largest_amount: Decimal


class AnalystReport(BaseModel):
    """Read-only query results plus the anomaly flags for one user."""

    user_id: int
    question: str
    sql: str
    rows: list[dict[str, object]]
    anomalies: list[SpendAnomaly]


class AdvisorDraft(BaseModel):
    """Hebrew narrative returned by the advisor model before formatting."""

    summary: str = Field(min_length=1)
    recommendations: list[str] = Field(default_factory=list)


class AdvisorResponse(BaseModel):
    """Advisor message ready to show to the user."""

    message_he: str
    summary: str
    recommendations: list[str]
