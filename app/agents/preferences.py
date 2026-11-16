"""Store an explicit categorization rule in ChromaDB."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.agents.ingestion import category_for_name, save_category_preference
from app.agents.supervisor import latest_user_text
from app.db.models import User
from app.graph.runtime import as_model, llm_from_config, preferences_from_config, session_scope
from app.graph.state import AgentState
from app.schemas.intent import PreferenceCommand

logger = logging.getLogger(__name__)


def preference_updater_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
    """Parse a standing rule and save it on the user_preferences collection."""
    user_text = latest_user_text(state)
    model = llm_from_config(config)
    prompt = (
        "Extract the merchant and the Hebrew category from this standing rule. "
        "The user is saying that this merchant should always use that category.\n"
        "User message:\n"
        f"{user_text}"
    )
    try:
        runnable = getattr(model, "with_structured_output")(PreferenceCommand)
        parsed = getattr(runnable, "invoke")([HumanMessage(content=prompt)])
    except Exception:
        logger.exception("Failed to parse a preference rule for user %s", state.get("user_id"))
        raise

    command = as_model(parsed, PreferenceCommand)
    with session_scope(config) as session:
        user = session.get(User, state["user_id"])
        if user is None:
            raise LookupError(f"User {state['user_id']} does not exist")
        category = category_for_name(session, user.id, command.category_name)
        session.commit()
        save_category_preference(
            preferences_from_config(config),
            user=user,
            merchant=command.merchant.strip(),
            category_name=category.name,
            category_slug=category.slug,
        )

    text = f"שמרתי כלל: עסקאות של {command.merchant.strip()} יסווגו כ{command.category_name.strip()}."
    return {"final_response": text, "messages": [AIMessage(content=text)], "pending_receipt": None}
