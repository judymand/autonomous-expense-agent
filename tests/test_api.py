"""Offline tests for the HTTP API and Telegram approval buttons."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.main import create_app
from app.api.telegram_bot import (
    build_approval_keyboard,
    decision_from_callback,
    on_callback,
    on_photo,
    on_text,
)
from app.db.models import Base, Category, ConversationThread, Transaction, User
from app.db.seed import CATEGORIES
from app.graph.workflow import build_workflow, open_sqlite_checkpointer
from app.schemas.analysis import AdvisorDraft, GeneratedSQL
from app.schemas.intent import ChatReply, IntentClassification, PreferenceCommand
from app.schemas.receipt import ReceiptExtraction
from app.services.assistant import AssistantRunner, SessionFactory


class _Runnable:
    def __init__(self, parent: ScriptedLLM, schema: type[object]) -> None:
        self._parent = parent
        self._schema = schema

    def invoke(self, messages: list[object]) -> object:
        content = getattr(messages[0], "content", "")
        if isinstance(content, list):
            prompt = " ".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        else:
            prompt = content if isinstance(content, str) else str(content)
        self._parent.schemas.append(self._schema)
        self._parent.prompts.append(prompt)
        return _respond(self._schema, prompt, content)


class ScriptedLLM:
    """Offline stand-in that never calls Gemini or Ollama."""

    def __init__(self) -> None:
        self.schemas: list[type[object]] = []
        self.prompts: list[str] = []

    def with_structured_output(self, schema: type[object]) -> _Runnable:
        return _Runnable(self, schema)


class _HashEmbedder:
    """Local embedder so preference writes do not download a model."""

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


class _DeadChroma:
    def heartbeat(self) -> int:
        raise RuntimeError("chroma is down")


class _Person:
    def __init__(self, user_id: int, full_name: str) -> None:
        self.id = user_id
        self.full_name = full_name
        self.username = None


class _Chat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class _Message:
    def __init__(
        self,
        text: str | None = None,
        photo: list[object] | None = None,
        caption: str | None = None,
    ) -> None:
        self.text = text
        self.photo = photo
        self.caption = caption
        self.document = None
        self.replies: list[tuple[str, object]] = []

    async def reply_text(self, text: str, reply_markup: object = None) -> None:
        self.replies.append((text, reply_markup))


class _Update:
    def __init__(self, message: _Message, *, chat_id: int, user_id: int) -> None:
        self.message = message
        self.effective_chat = _Chat(chat_id)
        self.effective_user = _Person(user_id, "דנה")
        self.callback_query = None


class _Query:
    def __init__(self, data: str) -> None:
        self.data = data
        self.message = _Message()
        self.edits: list[tuple[str, object]] = []

    async def answer(self) -> None:
        return None

    async def edit_message_text(self, text: str, reply_markup: object = None) -> None:
        self.edits.append((text, reply_markup))


class _CallbackUpdate:
    def __init__(self, query: _Query, *, chat_id: int, user_id: int) -> None:
        self.callback_query = query
        self.effective_chat = _Chat(chat_id)
        self.effective_user = _Person(user_id, "דנה")
        self.message = None


class _Context:
    def __init__(self, runner: AssistantRunner, sessions: SessionFactory) -> None:
        self.bot_data: dict[str, object] = {"runner": runner, "session_factory": sessions}


class _TelegramFile:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def get_file(self) -> _TelegramFile:
        return self

    async def download_to_drive(self, custom_path: str) -> None:
        Path(custom_path).write_bytes(self._payload)


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255 for byte in digest[:8]]


def _section(prompt: str, marker: str) -> str:
    if marker not in prompt:
        return prompt
    return prompt.split(marker, 1)[1]


def _respond(schema: type[object], prompt: str, content: object) -> object:
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
        if isinstance(content, list):
            amount = Decimal("40")
        else:
            match = re.search(r"(\d+(?:\.\d+)?)", _section(prompt, "Expense text:"))
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


@contextmanager
def _world(tmp_path: Path) -> Iterator[tuple[TestClient, int, SessionFactory]]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'expenses.sqlite3'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def sessions() -> Iterator[Session]:
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    with sessions() as session:
        user = User(telegram_id=501, display_name="בדיקה", preferred_currency="ILS")
        session.add(user)
        session.flush()
        session.add_all(
            [
                Category(user_id=user.id, slug="groceries", name="סופר ומכולת"),
                Category(user_id=user.id, slug="restaurants", name="מסעדות ובתי קפה"),
            ]
        )
        user_id = user.id

    chroma = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    preferences: Collection = chroma.get_or_create_collection(
        name="user_preferences",
        embedding_function=_HashEmbedder(),
        metadata={"hnsw:space": "cosine"},
    )
    graph = build_workflow(open_sqlite_checkpointer(tmp_path / "checkpoints.sqlite3"))
    runner = AssistantRunner(
        graph,
        session_factory=sessions,
        llm=ScriptedLLM(),
        preferences=preferences,
    )
    application = create_app(runner, sqlite_engine=engine, chroma_client=chroma)
    with TestClient(application) as client:
        yield client, user_id, sessions


def _count_transactions(sessions: SessionFactory) -> int:
    with sessions() as session:
        return int(session.scalar(select(func.count()).select_from(Transaction)) or 0)


def test_env_example_lists_langsmith_and_telegram() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")
    for name in (
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "TELEGRAM_BOT_TOKEN",
    ):
        assert name in text


def test_health_reports_sqlite_and_chroma(tmp_path: Path) -> None:
    with _world(tmp_path) as (client, _user_id, _sessions):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "sqlite": "ok", "chroma": "ok"}


def test_health_is_degraded_when_chroma_is_down(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'health.sqlite3'}")
    with engine.connect() as connection:
        connection.exec_driver_sql("SELECT 1")
    graph = build_workflow(open_sqlite_checkpointer(tmp_path / "checkpoints.sqlite3"))
    application = create_app(
        AssistantRunner(graph),
        sqlite_engine=engine,
        chroma_client=_DeadChroma(),
    )
    with TestClient(application) as client:
        response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["sqlite"] == "ok"
    assert body["chroma"] == "error"


def test_chat_answers_a_budget_question(tmp_path: Path) -> None:
    with _world(tmp_path) as (client, user_id, _sessions):
        response = client.post(
            "/chat",
            json={
                "user_id": user_id,
                "thread_id": "budget",
                "message": "כמה הוצאתי על סופר החודש?",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert "אין חריגה בולטת" in body["message"]
    assert body["pending_receipt"] is None


def test_chat_interrupts_a_large_expense_and_resume_corrects_category(tmp_path: Path) -> None:
    with _world(tmp_path) as (client, user_id, sessions):
        paused = client.post(
            "/chat",
            json={"user_id": user_id, "thread_id": "hitl", "message": "שילמתי 640 ש״ח בשופרסל"},
        )
        assert paused.status_code == 200
        body = paused.json()
        assert body["status"] == "interrupted"
        assert body["pending_receipt"]["merchant_name"] == "שופרסל"
        assert body["pending_receipt"]["total_amount"] == "640.00"
        assert body["pending_receipt"]["suggested_category"] == "סופר ומכולת"
        assert "amount_over_500" in body["interrupt"]["reasons"]
        assert "הסכום גבוה מ-500" in body["message"]
        assert _count_transactions(sessions) == 0

        blocked = client.post(
            "/chat",
            json={"user_id": user_id, "thread_id": "hitl", "message": "שלום"},
        )
        assert blocked.status_code == 409

        resumed = client.post(
            "/chat/resume",
            json={
                "user_id": user_id,
                "thread_id": "hitl",
                "approved": True,
                "category": "מסעדות ובתי קפה",
            },
        )
        assert resumed.status_code == 200
        saved = resumed.json()
        assert saved["status"] == "completed"
        assert "מסעדות ובתי קפה" in saved["message"]
        assert "מעכשיו אסווג" in saved["message"]

        with sessions() as session:
            stored = session.scalar(select(Transaction))
            assert stored is not None
            assert stored.amount == Decimal("640.00")
            assert stored.merchant == "שופרסל"
            assert stored.category.name == "מסעדות ובתי קפה"


def test_resume_reject_does_not_store_the_expense(tmp_path: Path) -> None:
    with _world(tmp_path) as (client, user_id, sessions):
        client.post(
            "/chat",
            json={"user_id": user_id, "thread_id": "reject", "message": "שילמתי 640 ש״ח בשופרסל"},
        )
        resumed = client.post(
            "/chat/resume",
            json={"user_id": user_id, "thread_id": "reject", "approved": False},
        )
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "completed"
        assert "לא נשמרה" in resumed.json()["message"]
        assert _count_transactions(sessions) == 0

        missing = client.post(
            "/chat/resume",
            json={"user_id": user_id, "thread_id": "reject", "approved": True},
        )
        assert missing.status_code == 409


def test_chat_upload_passes_the_receipt_to_the_graph(tmp_path: Path) -> None:
    image = tmp_path / "receipt.png"
    Image.new("RGB", (8, 8), "white").save(image, format="PNG")
    with _world(tmp_path) as (client, user_id, sessions):
        response = client.post(
            "/chat",
            data={"user_id": str(user_id), "thread_id": "receipt", "message": "קבלה"},
            files={"file": ("receipt.png", image.read_bytes(), "image/png")},
        )
        rejected = client.post(
            "/chat",
            data={"user_id": str(user_id), "thread_id": "notes", "message": "הערה"},
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert "שמרתי הוצאה של 40.00" in response.json()["message"]
    assert _count_transactions(sessions) == 1
    assert rejected.status_code == 415


def test_chat_unknown_user_is_not_found(tmp_path: Path) -> None:
    with _world(tmp_path) as (client, user_id, _sessions):
        response = client.post(
            "/chat",
            json={"user_id": user_id + 50, "thread_id": "missing", "message": "שלום"},
        )
    assert response.status_code == 404


def test_approval_keyboard_offers_approve_reject_and_categories() -> None:
    categories = [("groceries", "סופר ומכולת"), ("restaurants", "מסעדות ובתי קפה")]
    markup = build_approval_keyboard(categories)
    flat = [button for row in markup.inline_keyboard for button in row]
    assert flat[0].text == "Approve ✅"
    assert flat[0].callback_data == "approve"
    assert flat[1].text == "Reject ❌"
    assert flat[1].callback_data == "reject"
    assert [button.text for button in flat[2:]] == ["סופר ומכולת", "מסעדות ובתי קפה"]
    assert [button.callback_data for button in flat[2:]] == ["cat:groceries", "cat:restaurants"]
    assert all(len(button.callback_data.encode("utf-8")) <= 64 for button in flat)
    assert decision_from_callback("approve", categories) == (True, None)
    assert decision_from_callback("reject", categories) == (False, None)
    assert decision_from_callback("cat:restaurants", categories) == (True, "מסעדות ובתי קפה")
    assert decision_from_callback("cat:missing", categories) is None


def test_telegram_thread_is_stored_once(tmp_path: Path) -> None:
    _client, _user_id, sessions = _open_only(tmp_path)
    from app.services.telegram_identity import ensure_telegram_thread, list_categories

    with sessions() as session:
        user_id, thread_id = ensure_telegram_thread(
            session,
            chat_id=77,
            telegram_user_id=77,
            display_name="דנה",
        )
        again_user, again_thread = ensure_telegram_thread(
            session,
            chat_id=77,
            telegram_user_id=77,
            display_name="דנה",
        )
        stored = session.scalar(select(ConversationThread).where(ConversationThread.telegram_chat_id == 77))
        pairs = list_categories(session, user_id)
    assert thread_id == "telegram-77"
    assert (again_user, again_thread) == (user_id, thread_id)
    assert stored is not None
    assert stored.user_id == user_id
    assert stored.thread_id == thread_id
    assert pairs == list(CATEGORIES)


def test_telegram_interrupt_buttons_resume_the_graph(tmp_path: Path) -> None:
    runner, sessions = _runner(tmp_path)
    context = _Context(runner, sessions)

    async def scenario() -> None:
        message = _Message("שילמתי 640 ש״ח בשופרסל")
        await on_text(_Update(message, chat_id=42, user_id=42), context)  # type: ignore[arg-type]
        assert message.replies
        text, keyboard = message.replies[-1]
        assert "640.00" in text
        assert keyboard is not None
        labels = [button.text for row in keyboard.inline_keyboard for button in row]
        assert labels[0] == "Approve ✅"
        assert "Reject ❌" in labels
        assert "מסעדות ובתי קפה" in labels

        follow_up = _Message("שלום")
        await on_text(_Update(follow_up, chat_id=42, user_id=42), context)  # type: ignore[arg-type]
        assert follow_up.replies[-1][0].startswith("יש הוצאה שממתינה")

        query = _Query("cat:restaurants")
        await on_callback(_CallbackUpdate(query, chat_id=42, user_id=42), context)  # type: ignore[arg-type]
        assert query.edits
        assert "מסעדות ובתי קפה" in query.edits[-1][0]
        assert query.edits[-1][1] is None

    asyncio.run(scenario())
    with sessions() as session:
        stored = session.scalar(select(Transaction))
        assert stored is not None
        assert stored.amount == Decimal("640.00")
        assert stored.category.name == "מסעדות ובתי קפה"
        thread = session.scalar(select(ConversationThread).where(ConversationThread.telegram_chat_id == 42))
        assert thread is not None
        assert thread.thread_id == "telegram-42"


def test_telegram_photo_is_downloaded_and_saved(tmp_path: Path) -> None:
    runner, sessions = _runner(tmp_path)
    context = _Context(runner, sessions)
    image = tmp_path / "receipt.png"
    Image.new("RGB", (8, 8), "white").save(image, format="PNG")

    async def scenario() -> None:
        message = _Message(photo=[_TelegramFile(image.read_bytes())], caption="קבלה מהסופר")
        await on_photo(_Update(message, chat_id=88, user_id=88), context)  # type: ignore[arg-type]
        assert message.replies
        assert "שמרתי הוצאה של 40.00" in message.replies[-1][0]
        assert message.replies[-1][1] is None

    asyncio.run(scenario())
    assert _count_transactions(sessions) == 1


def test_telegram_reject_button_drops_the_expense(tmp_path: Path) -> None:
    runner, sessions = _runner(tmp_path)
    context = _Context(runner, sessions)

    async def scenario() -> None:
        message = _Message("שילמתי 640 ש״ח בשופרסל")
        await on_text(_Update(message, chat_id=9, user_id=9), context)  # type: ignore[arg-type]
        query = _Query("reject")
        await on_callback(_CallbackUpdate(query, chat_id=9, user_id=9), context)  # type: ignore[arg-type]
        assert "לא נשמרה" in query.edits[-1][0]

    asyncio.run(scenario())
    assert _count_transactions(sessions) == 0


def _runner(tmp_path: Path) -> tuple[AssistantRunner, SessionFactory]:
    _client, _user_id, sessions = _open_only(tmp_path)
    preferences = _preferences(tmp_path / "chroma")
    graph = build_workflow(open_sqlite_checkpointer(tmp_path / "checkpoints.sqlite3"))
    runner = AssistantRunner(
        graph,
        session_factory=sessions,
        llm=ScriptedLLM(),
        preferences=preferences,
    )
    return runner, sessions


def _open_only(tmp_path: Path) -> tuple[None, None, SessionFactory]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'expenses.sqlite3'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def sessions() -> Iterator[Session]:
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return None, None, sessions


def _preferences(path: Path) -> Collection:
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name="user_preferences",
        embedding_function=_HashEmbedder(),
        metadata={"hnsw:space": "cosine"},
    )
