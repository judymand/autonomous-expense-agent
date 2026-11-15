"""Environment-backed paths for the relational database and ChromaDB."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Local persistence settings loaded from the environment or `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(default="sqlite:///data/expenses.sqlite3")
    chroma_persist_dir: str = Field(default="data/chroma")

    def resolved_database_url(self) -> str:
        """Return the database URL with relative SQLite paths anchored at the project root."""
        prefix = "sqlite:///"
        url = self.database_url
        if not url.startswith(prefix):
            return url

        location = url[len(prefix) :]
        if location == ":memory:" or location.startswith("/"):
            return url

        absolute = (PROJECT_ROOT / location).resolve()
        return f"{prefix}{absolute}"

    def resolved_chroma_dir(self) -> Path:
        """Return the absolute ChromaDB persistence directory."""
        path = Path(self.chroma_persist_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
