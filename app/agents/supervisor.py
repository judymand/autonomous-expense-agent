"""Supervisor routing and the general-chat fallback."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.graph.runtime import as_model, llm_from_config
from app.graph.state import AgentState
from app.schemas.intent import ChatReply, IntentClassification

logger = logging.getLogger(__name__)


def latest_user_text(state: AgentState) -> str:
    """Return the text of the latest human message."""
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) != "human":
            continue
        return _content_text(getattr(message, "content", ""))
    return ""


def supervisor_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
    """Classify the latest user message into one graph branch."""
    user_text = latest_user_text(state)
    if not user_text.strip():
        raise ValueError("A user message is required")

    model = llm_from_config(config)
    prompt = (
        "Classify the user message into exactly one intent.\n"
        "ingest_expense: a receipt image or PDF, or a free-text expense such as a payment amount and merchant.\n"
        "analyze_finances: a question about spending, budgets, totals, or unusual expenses.\n"
        "update_preference: an explicit standing rule, such as always categorizing one merchant as a category.\n"
        "general_chat: greetings or anything that is not an expense, analysis, or preference rule.\n"
        "User message:\n"
        f"{user_text}"
    )
    try:
        runnable = getattr(model, "with_structured_output")(IntentClassification)
        parsed = getattr(runnable, "invoke")([HumanMessage(content=prompt)])
    except Exception:
        logger.exception("Supervisor failed to classify a message for user %s", state.get("user_id"))
        raise

    decision = as_model(parsed, IntentClassification)
    logger.info("Supervisor routed user %s to %s", state.get("user_id"), decision.intent)
    return {"intent": decision.intent}


def general_chat_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
    """Answer a message that does not need the expense tools."""
    user_text = latest_user_text(state)
    model = llm_from_config(config)
    prompt = (
        "You are a personal expense assistant. Reply in natural, concise Hebrew. "
        "Invite the user to send a receipt, describe an expense, or ask about spending. "
        "Do not invent transactions.\n"
        "User message:\n"
        f"{user_text}"
    )
    try:
        runnable = getattr(model, "with_structured_output")(ChatReply)
        parsed = getattr(runnable, "invoke")([HumanMessage(content=prompt)])
    except Exception:
        logger.exception("General chat failed for user %s", state.get("user_id"))
        raise

    reply = as_model(parsed, ChatReply)
    text = reply.message_he.strip()
    return {"final_response": text, "messages": [AIMessage(content=text)]}


def route_intent(state: AgentState) -> str:
    """Return the intent field written by the supervisor."""
    intent = state.get("intent")
    if intent not in {"ingest_expense", "analyze_finances", "update_preference", "general_chat"}:
        raise ValueError(f"Unknown intent: {intent}")
    return str(intent)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)
