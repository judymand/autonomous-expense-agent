"""Run receipt ingestion against one local image or PDF.

Example:
    .venv/bin/python scripts/extract_receipt.py data/sample_receipts/receipt.jpg --user-id 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.db.session import get_session, init_db
from app.services.ingestion import ingest_receipt

logger = logging.getLogger(__name__)


def main() -> None:
    """Extract one receipt with the configured vision model and print the JSON result."""
    parser = argparse.ArgumentParser(description="Extract a receipt image or PDF.")
    parser.add_argument("receipt", type=Path, help="Path to a receipt image or PDF")
    parser.add_argument("--user-id", type=int, default=1, help="Database user id to check for duplicates")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    init_db()
    try:
        with get_session() as session:
            result = ingest_receipt(args.receipt, user_id=args.user_id, session=session)
    except Exception:
        logger.exception("Receipt ingestion failed")
        raise SystemExit(1) from None

    logger.info("%s", json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
