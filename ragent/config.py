"""Paths and constants for RAGent."""

from pathlib import Path

BASE_DIR = Path.home() / ".ragent"
QDRANT_DIR = BASE_DIR / "qdrant_storage"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "ragent.log"
LLM_LOG_FILE = LOG_DIR / "llm_server.log"
RAGENT_SERVER_LOG_FILE = LOG_DIR / "ragent_server.log"
VECTORDB_LOG_FILE = LOG_DIR / "vectordb_server.log"
FAILED_CHUNKS_FILE = BASE_DIR / "failed_chunks.jsonl"

RAGENT_SERVER_HOST = "127.0.0.1"
RAGENT_SERVER_PORT = 8765
RAGENT_SERVER_URL = f"http://{RAGENT_SERVER_HOST}:{RAGENT_SERVER_PORT}"

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333

LLM_REPO_ID = "unsloth/Qwen3.5-9B-GGUF"
LLM_FILENAME = "Qwen3.5-9B-Q4_K_M.gguf"
LLM_SERVER_HOST = "127.0.0.1"
LLM_SERVER_PORT = 8000
LLM_API_BASE_URL = f"http://{LLM_SERVER_HOST}:{LLM_SERVER_PORT}/v1"

SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
DENSE_EMBEDDING_REPO_ID = "Qwen/Qwen3-Embedding-0.6B-GGUF"
DENSE_EMBEDDING_FILENAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"
DENSE_EMBEDDING_N_CTX = 8192
DENSE_EMBEDDING_N_SEQ_MAX = 4
DENSE_EMBEDDING_N_CTX_SEQ = DENSE_EMBEDDING_N_CTX // DENSE_EMBEDDING_N_SEQ_MAX  # 2048 tokens per sequence
SHORT_DENSE_SIZE = 256
LONG_DENSE_SIZE = 1024

MAX_CHUNK_SIZE = 1000

RERANKING_MODEL = "BAAI/bge-reranker-v2-m3"

ENABLE_CONTEXTUAL_RETRIEVAL = False

def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)