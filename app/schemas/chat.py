"""HTTP payloads for a chat turn and an approval resume."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Identity and text for one user turn. A receipt file is sent separately."""

    user_id: int = Field(gt=0)
    thread_id: str = Field(min_length=1, max_length=128)
    message: str = Field(default="", max_length=4000)

    @field_validator("thread_id", "message")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("thread_id")
    @classmethod
    def _single_token(cls, value: str) -> str:
        if not value or any(char.isspace() for char in value):
            raise ValueError("thread_id must be a single token")
        return value


class ResumeRequest(BaseModel):
    """Approve, reject, or correct the category of a paused expense."""

    user_id: int = Field(gt=0)
    thread_id: str = Field(min_length=1, max_length=128)
    approved: bool
    category: str | None = Field(default=None, max_length=80)

    @field_validator("thread_id")
    @classmethod
    def _single_token(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(char.isspace() for char in cleaned):
            raise ValueError("thread_id must be a single token")
        return cleaned

    @field_validator("category")
    @classmethod
    def _blank_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class TurnTrace(BaseModel):
    """Optional debug view of the nodes and tools used in one turn.

    Every field is optional so callers that ignore tracing stay compatible.
    """

    visited_nodes: list[str] = Field(default_factory=list)
    generated_sql: str | None = None
    matched_rule: dict[str, Any] | None = None


class ChatTurnResponse(BaseModel):
    """Either a finished Hebrew reply or a paused expense waiting for a person."""

    status: Literal["completed", "interrupted"]
    user_id: int
    thread_id: str
    message: str
    pending_receipt: dict[str, Any] | None = None
    interrupt: dict[str, Any] | None = None
    trace: TurnTrace | None = None


class HealthStatus(BaseModel):
    """Connectivity of the relational database and the vector store."""

    status: Literal["ok", "degraded"]
    sqlite: Literal["ok", "error"]
    chroma: Literal["ok", "error"]
