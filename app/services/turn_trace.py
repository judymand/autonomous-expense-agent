"""Build the optional per-turn trace from LangGraph checkpoints."""

from __future__ import annotations

import logging
from typing import Any

from app.schemas.chat import TurnTrace

logger = logging.getLogger(__name__)

_EXPENSE_NODES = {"ingestion", "approval_gate", "commit_expense"}
_HIDDEN_NODES = {"__start__", "__end__", "__interrupt__"}


def build_turn_trace(
    graph: object,
    thread_id: str,
    before_checkpoint_id: str | None,
    values: dict[str, Any],
) -> TurnTrace:
    """Describe the nodes that ran after ``before_checkpoint_id``."""
    visited = _visited_nodes(graph, thread_id, before_checkpoint_id)
    return TurnTrace(
        visited_nodes=visited,
        generated_sql=_generated_sql(values, visited),
        matched_rule=_matched_rule(values, visited),
    )


def checkpoint_id(snapshot: object) -> str | None:
    """Return the checkpoint id stored on a state snapshot."""
    config = getattr(snapshot, "config", None)
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        return None
    value = configurable.get("checkpoint_id")
    if isinstance(value, str) and value:
        return value
    return None


def _visited_nodes(graph: object, thread_id: str, before_checkpoint_id: str | None) -> list[str]:
    history = getattr(graph, "get_state_history", None)
    if not callable(history):
        return []
    try:
        snapshots = list(history({"configurable": {"thread_id": thread_id}}))
    except Exception:
        logger.exception("Could not read graph history for thread %s", thread_id)
        return []

    chunks: list[list[str]] = []
    for snapshot in snapshots:
        if before_checkpoint_id is not None and checkpoint_id(snapshot) == before_checkpoint_id:
            # Resume updates the interrupted checkpoint in place. Keep the node
            # only when that task now has a result, and do not walk into older turns.
            chunks.append(_finished_nodes(snapshot))
            break
        chunks.append(_nodes_in_snapshot(snapshot))

    ordered: list[str] = []
    for names in reversed(chunks):
        for name in names:
            if name in _HIDDEN_NODES or name.startswith("__"):
                continue
            if name not in ordered:
                ordered.append(name)
    return ordered


def _finished_nodes(snapshot: object) -> list[str]:
    names: list[str] = []
    for task in getattr(snapshot, "tasks", ()) or ():
        task_name = getattr(task, "name", None)
        if not isinstance(task_name, str):
            continue
        if getattr(task, "result", None) is not None or getattr(task, "error", None) is not None:
            names.append(task_name)
    return names


def _nodes_in_snapshot(snapshot: object) -> list[str]:
    """Node names for tasks that finished or paused in this checkpoint.

    LangGraph 1.2 stores the node on ``tasks`` and leaves ``metadata['writes']``
    empty. A task with a result has finished. A task with interrupts has paused.
    """
    names: list[str] = []
    for task in getattr(snapshot, "tasks", ()) or ():
        task_name = getattr(task, "name", None)
        if not isinstance(task_name, str):
            continue
        finished = getattr(task, "result", None) is not None or getattr(task, "error", None) is not None
        paused = bool(getattr(task, "interrupts", ()))
        if finished or paused:
            names.append(task_name)

    metadata = getattr(snapshot, "metadata", None)
    writes = metadata.get("writes") if isinstance(metadata, dict) else None
    if isinstance(writes, dict):
        names.extend(str(name) for name in writes if isinstance(name, str))
    return names


def _generated_sql(values: dict[str, Any], visited: list[str]) -> str | None:
    if "analyst" not in visited:
        return None
    payload = values.get("sql_result")
    if not isinstance(payload, dict):
        return None
    statement = payload.get("sql")
    if isinstance(statement, str) and statement.strip():
        return statement.strip()
    return None


def _matched_rule(values: dict[str, Any], visited: list[str]) -> dict[str, Any] | None:
    if not _EXPENSE_NODES.intersection(visited):
        return None
    pending = values.get("pending_receipt")
    if not isinstance(pending, dict):
        return None
    rule = pending.get("matched_rule")
    if isinstance(rule, dict):
        return dict(rule)
    return None
