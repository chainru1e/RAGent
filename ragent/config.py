"""Paths and constants for RAGent."""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = Path.home() / ".ragent"
QDRANT_DIR = BASE_DIR / "qdrant_storage"
LOG_FILE = BASE_DIR / "ragent.log"

EMBEDDING_MODEL = "BAAI/bge-m3"
SHORT_DENSE_SIZE = 256
LONG_DENSE_SIZE = 1024

RERANKING_MODEL = "BAAI/bge-reranker-v2-m3"

def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
