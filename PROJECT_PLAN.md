# Autonomous Financial Assistant - Architecture & Execution Plan

## 1. Project Overview
A multi-agent conversational expense and budget manager built with Python, LangGraph, FastAPI, SQLite/PostgreSQL, and ChromaDB. Interfaces via Telegram Bot (and optional Web UI).

## 2. Directory Structure
```text
├── app/
│   ├── agents/          # LangGraph nodes (router, ingestion, analyst, advisor)
│   ├── graph/           # State definition, graph builder, checkpointer setup
│   ├── db/              # SQLAlchemy models, session, seed scripts, vector store
│   ├── schemas/         # Pydantic models for structured LLM outputs & API
│   ├── services/        # OCR/Vision wrapper, duplicate detector, SQL executor
│   └── api/             # FastAPI endpoints & Telegram webhook/polling handler
├── data/                # Local SQLite DB, ChromaDB persistence, sample receipts
├── tests/               # Pytest unit & integration tests
├── .env.example
└── requirements.txt