"""SQLAlchemy engine and session lifecycle."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, Connection, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base
from app.db.settings import get_settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _set_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine() -> Engine:
    """Create the shared engine on first use and reuse it afterward."""
    global _engine
    if _engine is not None:
        return _engine

    url = get_settings().resolved_database_url()
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(url, connect_args=connect_args)
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _set_sqlite_foreign_keys)

    _engine = engine
    logger.info("Database engine ready")
    return engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the shared session factory bound to the application engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _session_factory


def init_db() -> None:
    """Create the SQLite parent directory and any missing tables."""
    url = get_settings().resolved_database_url()
    prefix = "sqlite:///"
    if url.startswith(prefix):
        location = url[len(prefix) :]
        if location != ":memory:":
            Path(location).parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        ensure_receipt_hash_column(connection)
    logger.info("Relational schema is ready")


def ensure_receipt_hash_column(connection: Connection) -> None:
    """Add ``receipt_hash`` to an existing SQLite database created before that column."""
    if connection.dialect.name != "sqlite":
        return

    table = connection.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'")
    ).scalar()
    if table is None:
        return

    columns = {row[1] for row in connection.execute(text("PRAGMA table_info(transactions)"))}
    if "receipt_hash" not in columns:
        connection.execute(text("ALTER TABLE transactions ADD COLUMN receipt_hash VARCHAR(64)"))
    connection.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_transactions_receipt_hash "
            "ON transactions (receipt_hash)"
        )
    )


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a session, committing on success and rolling back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.error("Database session rolled back")
        raise
    finally:
        session.close()
