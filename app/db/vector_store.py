"""Persistent ChromaDB client for long-term user preference memory."""

import logging

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.db.settings import get_settings

logger = logging.getLogger(__name__)

USER_PREFERENCES_COLLECTION = "user_preferences"


def get_chroma_client() -> ClientAPI:
    """Open the on-disk ChromaDB client, creating the directory when needed."""
    persist_dir = get_settings().resolved_chroma_dir()
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    logger.info("ChromaDB client opened at %s", persist_dir)
    return client


def get_user_preferences_collection(client: ClientAPI | None = None) -> Collection:
    """Return the collection that stores free-text user spending preferences."""
    chroma = client if client is not None else get_chroma_client()
    return chroma.get_or_create_collection(
        name=USER_PREFERENCES_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
