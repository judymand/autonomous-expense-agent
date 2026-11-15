"""Offline tests for the LLM factory and receipt ingestion pipeline."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Category, Transaction, User
from app.db.receipt_hash import compute_receipt_hash
from app.schemas.receipt import ReceiptExtraction
from app.services.ingestion import ingest_receipt
from app.services.llm import LLMSettings, get_chat_model

PNG = Path("receipt.png")


class _Runnable:
    def __init__(self, payload: ReceiptExtraction | Exception, sink: dict[str, object]) -> None:
        self._payload = payload
        self._sink = sink

    def invoke(self, messages: object) -> ReceiptExtraction:
        self._sink["messages"] = messages
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeLLM:
    """Stand-in vision model. Records the prompt and returns a canned extraction."""

    def __init__(self, payload: ReceiptExtraction | Exception) -> None:
        self.payload = payload
        self.sink: dict[str, object] = {}

    def with_structured_output(self, schema: type[object]) -> _Runnable:
        assert schema is ReceiptExtraction
        return _Runnable(self.payload, self.sink)


class FakePreferences:
    """In-memory preference query that does not open Chroma or download embeddings."""

    def __init__(self, result: dict[str, object] | Exception, count: int = 1) -> None:
        self._result = result
        self._count = count
        self.kwargs: dict[str, object] | None = None

    def count(self) -> int:
        return self._count

    def query(self, **kwargs: object) -> dict[str, object]:
        self.kwargs = kwargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _settings(**overrides: object) -> LLMSettings:
    return LLMSettings(_env_file=None, **overrides)


def _png(path: Path) -> Path:
    Image.new("RGB", (8, 8), "white").save(path, format="PNG")
    return path


def _session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return factory()


def _user(session: Session, telegram_id: int = 55) -> User:
    user = User(telegram_id=telegram_id, display_name="בדיקה", preferred_currency="ILS")
    session.add(user)
    session.flush()
    session.add(Category(user_id=user.id, slug="groceries", name="סופר ומכולת"))
    session.add(Category(user_id=user.id, slug="restaurants", name="מסעדות ובתי קפה"))
    session.flush()
    return user


def _extraction(**overrides: object) -> ReceiptExtraction:
    payload: dict[str, object] = {
        "merchant_name": "שופרסל",
        "transaction_date": date(2026, 7, 18),
        "total_amount": Decimal("4750.40"),
        "currency": "₪",
        "items": ["חלב", "לחם"],
        "suggested_category": "קניות",
        "confidence_score": 0.62,
    }
    payload.update(overrides)
    return ReceiptExtraction.model_validate(payload)


def test_google_factory_uses_gemini_flash() -> None:
    model = get_chat_model(_settings(llm_provider="google", google_api_key="test-not-real"))
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert model.model == "gemini-3.8-flash"


def test_ollama_factory_uses_local_model() -> None:
    model = get_chat_model(
        _settings(
            llm_provider="ollama",
            llm_model="gemini-3.8-flash",
            ollama_model="llava",
            ollama_base_url="http://127.0.0.1:11434",
        )
    )
    assert type(model).__name__ == "ChatOllama"
    assert model.model == "llava"


def test_google_factory_requires_an_api_key() -> None:
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        get_chat_model(_settings(llm_provider="google", google_api_key=None))


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(llm_provider="anthropic")


def test_receipt_schema_accepts_string_items() -> None:
    extraction = _extraction()
    assert extraction.currency == "₪"
    assert [item.name for item in extraction.items] == ["חלב", "לחם"]


def test_hash_uses_the_jerusalem_calendar_date() -> None:
    late_utc = datetime(2026, 7, 17, 22, 30, tzinfo=timezone.utc)
    from_timestamp = compute_receipt_hash("שופרסל", late_utc, Decimal("10"), "ILS")
    from_date = compute_receipt_hash("  שופרסל ", date(2026, 7, 18), Decimal("10.00"), "ils")
    assert from_timestamp == from_date


def test_preference_rule_overrides_category(tmp_path: Path) -> None:
    session = _session()
    user = _user(session)
    preferences = FakePreferences(
        {
            "metadatas": [[{"topic": "groceries", "merchant": "שופרסל", "telegram_id": 55}]],
            "distances": [[0.2]],
        }
    )
    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=user.id,
        session=session,
        llm=FakeLLM(_extraction()),
        preferences=preferences,
    )
    assert result.preference_applied is True
    assert result.extraction.suggested_category == "סופר ומכולת"
    assert result.extraction.confidence_score == 1.0
    assert result.extraction.currency == "ILS"
    assert result.is_duplicate is False
    assert preferences.kwargs is not None
    assert preferences.kwargs["where"] == {"telegram_id": 55}


def test_semantic_preference_overrides_when_close(tmp_path: Path) -> None:
    session = _session()
    user = _user(session)
    preferences = FakePreferences(
        {
            "metadatas": [[{"topic": "restaurants"}]],
            "distances": [[0.2]],
        }
    )
    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=user.id,
        session=session,
        llm=FakeLLM(_extraction(suggested_category="קניות", confidence_score=0.4)),
        preferences=preferences,
    )
    assert result.extraction.suggested_category == "מסעדות ובתי קפה"
    assert result.extraction.confidence_score == 1.0


def test_merchant_rule_wins_over_a_closer_semantic_match(tmp_path: Path) -> None:
    session = _session()
    user = _user(session)
    preferences = FakePreferences(
        {
            "metadatas": [[
                {"topic": "restaurants"},
                {"topic": "groceries", "merchant": "שופרסל"},
            ]],
            "distances": [[0.2, 0.15]],
        }
    )
    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=user.id,
        session=session,
        llm=FakeLLM(_extraction()),
        preferences=preferences,
    )
    assert result.extraction.suggested_category == "סופר ומכולת"
    assert result.extraction.confidence_score == 1.0


def test_existing_sqlite_file_gains_receipt_hash_column(tmp_path: Path) -> None:
    from sqlalchemy import text

    from app.db.session import ensure_receipt_hash_column

    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.sqlite3'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE transactions ("
                "id INTEGER PRIMARY KEY, user_id INTEGER, merchant VARCHAR(160), "
                "amount NUMERIC(12, 2), currency VARCHAR(3), transacted_at DATETIME)"
            )
        )
        ensure_receipt_hash_column(connection)
        columns = {row[1] for row in connection.execute(text("PRAGMA table_info(transactions)"))}
    assert "receipt_hash" in columns


def test_generic_merchant_does_not_match_preference_rules(tmp_path: Path) -> None:
    session = _session()
    user = _user(session)
    preferences = FakePreferences(
        {
            "metadatas": [[{"topic": "groceries", "merchant": 'שם החברה בע"מ'}]],
            "distances": [[0.05]],
        }
    )
    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=user.id,
        session=session,
        llm=FakeLLM(_extraction(merchant_name='שם החברה בע"מ')),
        preferences=preferences,
    )
    assert result.preference_applied is False
    assert result.extraction.suggested_category == "קניות"
    assert result.extraction.confidence_score == 0.62
    assert preferences.kwargs is None


def test_weak_distance_does_not_override_even_with_the_same_merchant(tmp_path: Path) -> None:
    session = _session()
    user = _user(session)
    preferences = FakePreferences(
        {
            "metadatas": [[{"topic": "groceries", "merchant": "שופרסל"}]],
            "distances": [[0.41]],
        }
    )
    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=user.id,
        session=session,
        llm=FakeLLM(_extraction()),
        preferences=preferences,
    )
    assert result.preference_applied is False
    assert result.extraction.suggested_category == "קניות"


def test_distant_preference_does_not_override(tmp_path: Path) -> None:
    session = _session()
    user = _user(session)
    preferences = FakePreferences(
        {
            "metadatas": [[{"topic": "restaurants"}]],
            "distances": [[0.8]],
        }
    )
    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=user.id,
        session=session,
        llm=FakeLLM(_extraction()),
        preferences=preferences,
    )
    assert result.preference_applied is False
    assert result.extraction.suggested_category == "קניות"
    assert result.extraction.confidence_score == 0.62


def test_preference_lookup_failure_keeps_the_model_category(tmp_path: Path) -> None:
    session = _session()
    user = _user(session)
    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=user.id,
        session=session,
        llm=FakeLLM(_extraction()),
        preferences=FakePreferences(RuntimeError("chroma down")),
    )
    assert result.extraction.suggested_category == "קניות"
    assert result.preference_applied is False


def test_duplicate_matches_stored_receipt_hash(tmp_path: Path) -> None:
    session = _session()
    user = _user(session)
    category_id = session.scalar(select(Category.id).where(Category.user_id == user.id))
    assert category_id is not None
    digest = compute_receipt_hash("שופרסל", date(2026, 7, 18), Decimal("4750.40"), "ILS")
    existing = Transaction(
        user_id=user.id,
        category_id=category_id,
        merchant="שופרסל",
        amount=Decimal("4750.40"),
        currency="ILS",
        receipt_hash=digest,
        transacted_at=datetime(2026, 7, 18, 16, 30, tzinfo=timezone.utc),
    )
    session.add(existing)
    session.flush()

    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=user.id,
        session=session,
        llm=FakeLLM(_extraction()),
        preferences=FakePreferences({}, count=0),
    )
    assert result.is_duplicate is True
    assert result.matched_transaction_id == existing.id
    assert result.receipt_hash == digest


def test_unhashed_history_still_matches_merchant_date_and_amount(tmp_path: Path) -> None:
    session = _session()
    user = _user(session)
    category_id = session.scalar(select(Category.id).where(Category.user_id == user.id))
    assert category_id is not None
    existing = Transaction(
        user_id=user.id,
        category_id=category_id,
        merchant="שופרסל",
        amount=Decimal("4750.40"),
        currency="ILS",
        receipt_hash=None,
        transacted_at=datetime(2026, 7, 18, 16, 30, tzinfo=timezone.utc),
    )
    session.add(existing)
    session.flush()

    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=user.id,
        session=session,
        llm=FakeLLM(_extraction()),
        preferences=FakePreferences({}, count=0),
    )
    assert result.is_duplicate is True
    assert existing.receipt_hash == result.receipt_hash


def test_same_receipt_for_another_user_is_not_a_duplicate(tmp_path: Path) -> None:
    session = _session()
    owner = _user(session, telegram_id=1)
    other = _user(session, telegram_id=2)
    category_id = session.scalar(select(Category.id).where(Category.user_id == owner.id))
    assert category_id is not None
    session.add(
        Transaction(
            user_id=owner.id,
            category_id=category_id,
            merchant="שופרסל",
            amount=Decimal("4750.40"),
            currency="ILS",
            receipt_hash=compute_receipt_hash("שופרסל", date(2026, 7, 18), Decimal("4750.40"), "ILS"),
            transacted_at=datetime(2026, 7, 18, 16, 30, tzinfo=timezone.utc),
        )
    )
    session.flush()
    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=other.id,
        session=session,
        llm=FakeLLM(_extraction()),
        preferences=FakePreferences({}, count=0),
    )
    assert result.is_duplicate is False


def test_pdf_pages_are_sent_to_the_model(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    session = _session()
    user = _user(session)
    document = pdfium.PdfDocument.new()
    document.new_page(200, 200)
    pdf_path = tmp_path / "receipt.pdf"
    document.save(pdf_path)
    document.close()

    llm = FakeLLM(_extraction())
    ingest_receipt(
        pdf_path,
        user_id=user.id,
        session=session,
        llm=llm,
        preferences=FakePreferences({}, count=0),
    )
    messages = llm.sink["messages"]
    assert isinstance(messages, list)
    content = messages[0].content
    assert content[0]["type"] == "text"
    assert "סופר ומכולת" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_unsupported_file_type_is_rejected(tmp_path: Path) -> None:
    session = _session()
    user = _user(session)
    source = tmp_path / "notes.txt"
    source.write_text("not a receipt", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        ingest_receipt(
            source,
            user_id=user.id,
            session=session,
            llm=FakeLLM(_extraction()),
            preferences=FakePreferences({}, count=0),
        )


def test_vision_errors_are_raised(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    session = _session()
    user = _user(session)
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="quota"):
            ingest_receipt(
                _png(tmp_path / "receipt.png"),
                user_id=user.id,
                session=session,
                llm=FakeLLM(RuntimeError("quota")),
                preferences=FakePreferences({}, count=0),
            )
    assert "Vision model failed" in caplog.text


def test_chroma_merchant_rule_overrides_category(tmp_path: Path) -> None:
    import chromadb

    session = _session()
    user = _user(session, telegram_id=55)
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("user_preferences", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=["rule-shufersal"],
        documents=["כאשר הקבלה היא משופרסל, סווג אותה כסופר ומכולת"],
        metadatas=[{"telegram_id": 55, "merchant": "שופרסל", "suggested_category": "סופר ומכולת"}],
    )
    result = ingest_receipt(
        _png(tmp_path / "receipt.png"),
        user_id=user.id,
        session=session,
        llm=FakeLLM(_extraction()),
        preferences=collection,
    )
    assert result.preference_applied is True
    assert result.extraction.suggested_category == "סופר ומכולת"
    assert result.extraction.confidence_score == 1.0
