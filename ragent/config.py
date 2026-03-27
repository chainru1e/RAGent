"""Paths and constants for RAGent."""

from pathlib import Path

BASE_DIR = Path.home() / ".ragent"
PENDING_DIR = BASE_DIR / "pending"
CHROMA_DIR = BASE_DIR / "chroma_db"
LOG_FILE = BASE_DIR / "ragent.log"

COLLECTION_QA = "qa_pairs"
COLLECTION_SUMMARIES = "session_summaries"


def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
