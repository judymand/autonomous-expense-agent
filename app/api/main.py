"""FastAPI application for health checks, chat turns, and approval resumes."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import ValidationError
from sqlalchemy import Engine
from starlette.datastructures import UploadFile

from app.db.session import init_db
from app.db.settings import PROJECT_ROOT
from app.graph.workflow import build_workflow
from app.schemas.chat import ChatRequest, ChatTurnResponse, HealthStatus, ResumeRequest
from app.services.assistant import AssistantError, AssistantRunner
from app.services.health import probe_chroma, probe_sqlite, probe_status

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
_RECEIPT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"}


def load_runtime_env() -> None:
    """Export `.env` and Streamlit secrets so LangChain and the model factory can see them.

    Pydantic settings read the file for their own fields only. LangChain looks
    up ``LANGCHAIN_TRACING_V2``, ``LANGCHAIN_API_KEY``, and ``LANGCHAIN_PROJECT``
    on the process environment. ``GOOGLE_API_KEY`` is also copied from
    ``st.secrets`` when the Streamlit runtime is already loaded.
    """
    from app.services.llm import prime_llm_environment

    prime_llm_environment()
    load_dotenv(PROJECT_ROOT / ".env", override=False)


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    if application.state.runner is None:
        load_runtime_env()
        init_db()
        application.state.runner = AssistantRunner(build_workflow())
        logger.info("Assistant graph is ready")
    yield


def create_app(
    runner: AssistantRunner | None = None,
    *,
    sqlite_engine: Engine | None = None,
    chroma_client: object | None = None,
) -> FastAPI:
    """Build the API. Tests pass a runner and local databases; production does not."""
    application = FastAPI(title="Autonomous Expense Assistant", lifespan=_lifespan)
    application.state.runner = runner
    application.state.sqlite_engine = sqlite_engine
    application.state.chroma_client = chroma_client

    @application.get("/health", response_model=HealthStatus)
    def health(response: Response) -> HealthStatus:
        sqlite_status = probe_status(
            lambda: probe_sqlite(application.state.sqlite_engine),
            "SQLite",
        )
        chroma_status = probe_status(
            lambda: probe_chroma(application.state.chroma_client),
            "ChromaDB",
        )
        healthy = sqlite_status == "ok" and chroma_status == "ok"
        response.status_code = 200 if healthy else 503
        return HealthStatus(
            status="ok" if healthy else "degraded",
            sqlite=sqlite_status,
            chroma=chroma_status,
        )

    @application.post("/chat", response_model=ChatTurnResponse)
    async def chat(request: Request) -> ChatTurnResponse:
        payload, upload = await _read_chat_request(request)
        if not payload.message and upload is None:
            raise HTTPException(status_code=422, detail="message or a receipt file is required")
        message = payload.message or "קבלה מצורפת"
        path: Path | None = None
        try:
            if upload is not None:
                path = await _store_upload(upload)
            turn = await asyncio.to_thread(
                _runner(request).chat,
                user_id=payload.user_id,
                thread_id=payload.thread_id,
                message=message,
                receipt_path=path,
            )
        except AssistantError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Chat turn failed for thread %s", payload.thread_id)
            raise HTTPException(status_code=500, detail="The assistant could not complete this turn") from exc
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
        return turn

    @application.post("/chat/resume", response_model=ChatTurnResponse)
    async def resume(body: ResumeRequest) -> ChatTurnResponse:
        try:
            return await asyncio.to_thread(
                _runner_from_app(application).resume,
                user_id=body.user_id,
                thread_id=body.thread_id,
                approved=body.approved,
                category=body.category,
            )
        except AssistantError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except Exception as exc:
            logger.exception("Resume failed for thread %s", body.thread_id)
            raise HTTPException(status_code=500, detail="The assistant could not resume this thread") from exc

    return application


def _runner(request: Request) -> AssistantRunner:
    return _runner_from_app(request.app)


def _runner_from_app(application: FastAPI) -> AssistantRunner:
    runner = application.state.runner
    if not isinstance(runner, AssistantRunner):
        raise HTTPException(status_code=503, detail="The assistant is not ready")
    return runner


async def _read_chat_request(request: Request) -> tuple[ChatRequest, UploadFile | None]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        try:
            payload = ChatRequest.model_validate(await request.json())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return payload, None

    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(status_code=415, detail="Use application/json or multipart/form-data")

    form = await request.form()
    try:
        payload = ChatRequest.model_validate(
            {
                "user_id": form.get("user_id"),
                "thread_id": form.get("thread_id"),
                "message": form.get("message") or "",
            }
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    upload = form.get("file")
    if upload is None or upload == "":
        return payload, None
    if not isinstance(upload, UploadFile):
        raise HTTPException(status_code=422, detail="file must be a receipt upload")
    return payload, upload


async def _store_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in _RECEIPT_SUFFIXES:
        raise HTTPException(status_code=415, detail="Receipt files must be an image or a PDF")
    try:
        payload = await upload.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await upload.close()
    if not payload:
        raise HTTPException(status_code=422, detail="Receipt file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Receipt file is too large")

    descriptor, name = tempfile.mkstemp(suffix=suffix)
    os.close(descriptor)
    path = Path(name)
    try:
        path.write_bytes(payload)
    except Exception:
        path.unlink(missing_ok=True)
        logger.exception("Could not store an uploaded receipt")
        raise
    return path


app = create_app()
