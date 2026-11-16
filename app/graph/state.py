"""Shared state for the expense assistant graph."""

from typing import Annotated, Any, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.schemas.intent import IntentName


class AgentState(TypedDict):
    """Conversation state passed between the supervisor and specialist nodes.

    ``messages`` accumulates the thread. The other fields are replaced by the
    node that owns them. ``pending_receipt`` is the expense waiting for a
    decision; it is not written to SQLite until the approval gate allows it.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: int
    intent: NotRequired[IntentName]
    pending_receipt: NotRequired[dict[str, Any] | None]
    sql_result: NotRequired[dict[str, Any] | None]
    anomalies: NotRequired[list[dict[str, Any]]]
    requires_approval: NotRequired[bool]
    approval_status: NotRequired[str | None]
    final_response: NotRequired[str]
