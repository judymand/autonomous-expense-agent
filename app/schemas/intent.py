"""Structured outputs for routing, preference rules, and general chat."""

from typing import Literal

from pydantic import BaseModel, Field

IntentName = Literal["ingest_expense", "analyze_finances", "update_preference", "general_chat"]


class IntentClassification(BaseModel):
    """Supervisor decision for the latest user message."""

    intent: IntentName


class PreferenceCommand(BaseModel):
    """An explicit categorization rule stated by the user."""

    merchant: str = Field(min_length=1)
    category_name: str = Field(min_length=1)


class ChatReply(BaseModel):
    """Short Hebrew reply for a message that is not an expense task."""

    message_he: str = Field(min_length=1)
