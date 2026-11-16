"""LangGraph workflow: supervisor, specialists, and the approval interrupt."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.agents.advisor import run_advisor
from app.agents.analyst import run_financial_analyst
from app.agents.ingestion import approval_gate, commit_expense_node, ingestion_node
from app.agents.preferences import preference_updater_node
from app.agents.supervisor import general_chat_node, route_intent, supervisor_node, latest_user_text
from app.db.settings import PROJECT_ROOT
from app.graph.runtime import llm_from_config, session_scope
from app.graph.state import AgentState
from app.schemas.analysis import AnalystReport, SpendAnomaly

CHECKPOINT_PATH = PROJECT_ROOT / "data" / "checkpoints.sqlite3"


def open_sqlite_checkpointer(path: Path | str | None = None) -> SqliteSaver:
    """Open a LangGraph ``SqliteSaver`` stored at ``data/checkpoints.sqlite3`` by default."""
    location = Path(path) if path is not None else CHECKPOINT_PATH
    if str(location) != ":memory:":
        location.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(location), check_same_thread=False)
    else:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
    return SqliteSaver(connection)


def build_workflow(checkpointer: SqliteSaver | None = None):
    """Compile the assistant graph with a SQLite checkpointer."""
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("ingestion", ingestion_node)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("commit_expense", commit_expense_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("advisor", advisor_node)
    builder.add_node("preference_updater", preference_updater_node)
    builder.add_node("general_chat", general_chat_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_intent,
        {
            "ingest_expense": "ingestion",
            "analyze_finances": "analyst",
            "update_preference": "preference_updater",
            "general_chat": "general_chat",
        },
    )
    builder.add_edge("ingestion", "approval_gate")
    builder.add_edge("approval_gate", "commit_expense")
    builder.add_edge("commit_expense", END)
    builder.add_edge("analyst", "advisor")
    builder.add_edge("advisor", END)
    builder.add_edge("preference_updater", END)
    builder.add_edge("general_chat", END)

    saver = checkpointer if checkpointer is not None else open_sqlite_checkpointer()
    return builder.compile(checkpointer=saver)


def analyst_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
    """Run a read-only question and attach the anomaly flags."""
    question = latest_user_text(state)
    with session_scope(config) as session:
        report = run_financial_analyst(
            question,
            user_id=state["user_id"],
            session=session,
            llm=llm_from_config(config),
        )
    return {
        "sql_result": {"sql": report.sql, "rows": report.rows},
        "anomalies": [anomaly.model_dump(mode="json") for anomaly in report.anomalies],
        "pending_receipt": None,
        "requires_approval": False,
    }


def advisor_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
    """Turn the analyst payload into the Hebrew reply stored on the state."""
    payload = state.get("sql_result") or {}
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    sql = payload.get("sql", "") if isinstance(payload, dict) else ""
    anomalies = [SpendAnomaly.model_validate(item) for item in state.get("anomalies") or []]
    report = AnalystReport(
        user_id=state["user_id"],
        question=latest_user_text(state),
        sql=str(sql),
        rows=list(rows) if isinstance(rows, list) else [],
        anomalies=anomalies,
    )
    response = run_advisor(report, llm=llm_from_config(config))
    return {
        "final_response": response.message_he,
        "messages": [AIMessage(content=response.message_he)],
    }
