"""Offline routing and human-in-the-loop tests for the expense graph."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.agents.ingestion import approval_reasons
from app.db.models import Base, Category, Transaction, User
from app.graph.workflow import build_workflow, open_sqlite_checkpointer
from app.schemas.analysis import AdvisorDraft, GeneratedSQL
from app.schemas.intent import ChatReply, IntentClassification, PreferenceCommand
from app.schemas.receipt import ReceiptExtraction


class _Runnable:
    def __init__(self, parent: ScriptedLLM, schema: type[object]) -> None:
        self._parent = parent
        self._schema = schema

    def invoke(self, messages: list[object]) -> object:
        content = getattr(messages[0], "content", "")
        prompt = content if isinstance(content, str) else str(content)
        self._parent.schemas.append(self._schema)
        self._parent.prompts.append(prompt)
        return _respond(self._schema, prompt)


class ScriptedLLM:
    """Offline stand-in that returns a schema-specific answer from the prompt text."""

    def __init__(self) -> None:
        self.schemas: list[type[object]] = []
        self.prompts: list[str] = []

    def with_structured_output(self, schema: type[object]) -> _Runnable:
        return _Runnable(self, schema)


class _HashEmbedder:
    """Tiny local embedder so preference tests never download a model."""

    def is_legacy(self) -> bool:
        return True

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [_vector(text) for text in input]

    @staticmethod
    def name() -> str:
        return "hash-test"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    @staticmethod
    def get_config() -> dict[str, str]:
        return {}

    @classmethod
    def build_from_config(cls, config: dict[str, str]) -> _HashEmbedder:
        del config
        return cls()


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255 for byte in digest[:8]]


def _section(prompt: str, marker: str) -> str:
    if marker not in prompt:
        return prompt
    return prompt.split(marker, 1)[1]


def _respond(schema: type[object], prompt: str) -> object:
    if schema is IntentClassification:
        message = _section(prompt, "User message:")
        if "תמיד" in message:
            intent = "update_preference"
        elif any(word in message for word in ("כמה", "תקציב", "חריג")):
            intent = "analyze_finances"
        elif message.strip().startswith("שלום"):
            intent = "general_chat"
        else:
            intent = "ingest_expense"
        return IntentClassification(intent=intent)

    if schema is ReceiptExtraction:
        expense = _section(prompt, "Expense text:")
        match = re.search(r"(\d+(?:\.\d+)?)", expense)
        amount = Decimal(match.group(1) if match else "40")
        return ReceiptExtraction(
            merchant_name="שופרסל",
            transaction_date=date(2026, 11, 15),
            total_amount=amount,
            currency="ILS",
            items=["לחם"],
            suggested_category="סופר ומכולת",
            confidence_score=0.93,
        )

    if schema is GeneratedSQL:
        match = re.search(r"user_id = (\d+)", prompt)
        user_id = match.group(1) if match else "1"
        return GeneratedSQL(sql=f"SELECT COUNT(*) AS n FROM transactions WHERE user_id = {user_id}")

    if schema is AdvisorDraft:
        return AdvisorDraft(summary="אין חריגה בולטת בהוצאות.", recommendations=["המשיכו לעקוב אחרי הסופר"])

    if schema is PreferenceCommand:
        return PreferenceCommand(merchant="שופרסל", category_name="סופר ומכולת")

    if schema is ChatReply:
        return ChatReply(message_he="אפשר לשלוח קבלה, לתאר הוצאה, או לשאול על התקציב.")

    raise AssertionError(f"Unexpected schema {schema}")


def _session(path: Path) -> Session:
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = factory()
    user = User(telegram_id=501, display_name="בדיקה", preferred_currency="ILS")
    session.add(user)
    session.flush()
    session.add_all(
        [
            Category(user_id=user.id, slug="groceries", name="סופר ומכולת"),
            Category(user_id=user.id, slug="restaurants", name="מסעדות ובתי קפה"),
        ]
    )
    session.commit()
    return session


def _preferences(path: Path) -> Collection:
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name="user_preferences",
        embedding_function=_HashEmbedder(),
        metadata={"hnsw:space": "cosine"},
    )


def _invoke(tmp_path: Path, session: Session, text: str, thread_id: str) -> tuple[object, dict[str, object], ScriptedLLM, Collection]:
    llm = ScriptedLLM()
    preferences = _preferences(tmp_path / "chroma")
    graph = build_workflow(open_sqlite_checkpointer(tmp_path / "checkpoints.sqlite3"))
    config: dict[str, object] = {
        "configurable": {
            "thread_id": thread_id,
            "llm": llm,
            "session": session,
            "preferences": preferences,
        }
    }
    result = graph.invoke(
        {"messages": [HumanMessage(content=text)], "user_id": session.scalar(select(User.id))},
        config,
    )
    return graph, result, llm, preferences


def test_approval_reasons_cover_amount_confidence_and_duplicates() -> None:
    base = {
        "total_amount": "40.00",
        "currency": "ILS",
        "confidence_score": 0.93,
        "is_duplicate": False,
    }
    assert approval_reasons(base) == []
    assert approval_reasons({**base, "total_amount": "500.00"}) == []
    assert approval_reasons({**base, "total_amount": "500.01"}) == ["amount_over_500"]
    assert approval_reasons({**base, "confidence_score": 0.80}) == []
    assert approval_reasons({**base, "confidence_score": 0.79}) == ["low_confidence"]
    assert approval_reasons({**base, "is_duplicate": True}) == ["duplicate"]


def test_routes_spending_question_to_analyst_and_advisor(tmp_path: Path) -> None:
    session = _session(tmp_path / "expenses.sqlite3")
    _graph, result, llm, _preferences_collection = _invoke(
        tmp_path,
        session,
        "כמה הוצאתי על סופר החודש?",
        "analyze",
    )
    assert result["intent"] == "analyze_finances"
    assert llm.schemas == [IntentClassification, GeneratedSQL, AdvisorDraft]
    assert result["sql_result"]["rows"] == [{"n": 0}]
    assert result["anomalies"] == []
    assert "אין חריגה בולטת" in result["final_response"]
    assert "• המשיכו לעקוב אחרי הסופר" in result["final_response"]
    session.close()


def test_routes_text_expense_and_saves_without_interrupt(tmp_path: Path) -> None:
    session = _session(tmp_path / "expenses.sqlite3")
    graph, result, llm, preferences = _invoke(tmp_path, session, "שילמתי 40 ש״ח בשופרסל על לחם", "ingest-small")
    assert result["intent"] == "ingest_expense"
    assert llm.schemas == [IntentClassification, ReceiptExtraction]
    assert result["requires_approval"] is False
    assert graph.get_state({"configurable": {"thread_id": "ingest-small"}}).next == ()
    assert "שמרתי הוצאה של 40.00" in result["final_response"]
    stored = session.scalar(select(Transaction))
    assert stored is not None
    assert stored.amount == Decimal("40.00")
    assert stored.category.name == "סופר ומכולת"
    assert preferences.count() == 0
    session.close()


def test_routes_preference_rule_into_chroma(tmp_path: Path) -> None:
    session = _session(tmp_path / "expenses.sqlite3")
    _graph, result, llm, preferences = _invoke(
        tmp_path,
        session,
        "תמיד תסווג את שופרסל כסופר ומכולת",
        "preference",
    )
    assert result["intent"] == "update_preference"
    assert llm.schemas == [IntentClassification, PreferenceCommand]
    assert "שמרתי כלל" in result["final_response"]
    found = preferences.get(include=["documents", "metadatas"])
    assert any("שופרסל" in document for document in found["documents"])
    assert any(metadata["suggested_category"] == "סופר ומכולת" for metadata in found["metadatas"])
    assert session.scalar(select(func.count()).select_from(Transaction)) == 0
    session.close()


def test_routes_general_chat(tmp_path: Path) -> None:
    session = _session(tmp_path / "expenses.sqlite3")
    _graph, result, llm, preferences = _invoke(tmp_path, session, "שלום, מה אתה יודע לעשות?", "chat")
    assert result["intent"] == "general_chat"
    assert llm.schemas == [IntentClassification, ChatReply]
    assert result["final_response"].startswith("אפשר לשלוח קבלה")
    assert preferences.count() == 0
    assert session.scalar(select(func.count()).select_from(Transaction)) == 0
    session.close()


def test_hitl_pauses_above_500_and_saves_on_resume(tmp_path: Path) -> None:
    session = _session(tmp_path / "expenses.sqlite3")
    llm = ScriptedLLM()
    preferences = _preferences(tmp_path / "chroma")
    graph = build_workflow(open_sqlite_checkpointer(tmp_path / "checkpoints.sqlite3"))
    user_id = session.scalar(select(User.id))
    config = {
        "configurable": {
            "thread_id": "hitl-640",
            "llm": llm,
            "session": session,
            "preferences": preferences,
        }
    }
    graph.invoke(
        {"messages": [HumanMessage(content="שילמתי 640 ש״ח בשופרסל")], "user_id": user_id},
        config,
    )

    paused = graph.get_state(config)
    assert paused.next == ("approval_gate",)
    assert paused.values["requires_approval"] is True
    assert paused.interrupts
    payload = paused.interrupts[0].value
    assert "amount_over_500" in payload["reasons"]
    assert Decimal(payload["total_amount"]) > Decimal("500")
    assert session.scalar(select(func.count()).select_from(Transaction)) == 0

    resumed = graph.invoke(
        Command(resume={"approved": True, "category": "מסעדות ובתי קפה"}),
        config,
    )
    assert resumed["approval_status"] == "approved"
    assert "מסעדות ובתי קפה" in resumed["final_response"]
    assert "מעכשיו אסווג" in resumed["final_response"]

    stored = session.scalar(select(Transaction))
    assert stored is not None
    assert stored.amount == Decimal("640.00")
    assert stored.merchant == "שופרסל"
    assert stored.category.name == "מסעדות ובתי קפה"
    found = preferences.get(include=["documents", "metadatas"])
    assert any("שופרסל" in document for document in found["documents"])
    assert any(metadata["suggested_category"] == "מסעדות ובתי קפה" for metadata in found["metadatas"])
    assert llm.schemas == [IntentClassification, ReceiptExtraction]
    session.close()
