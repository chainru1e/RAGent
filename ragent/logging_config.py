"""Centralized logging configuration for RAGent.

Logger hierarchy:
  ragent              → logs/ragent.log         (all components via propagation)
  ragent.server       → logs/ragent_server.log
  ragent.vectordb     → logs/vectordb_server.log
  ragent.llm          → logs/llm_server.log
"""

import logging

from ragent.config import (
    LOG_DIR,
    LOG_FILE,
    LLM_LOG_FILE,
    RAGENT_SERVER_LOG_FILE,
    VECTORDB_LOG_FILE,
)

_configured = False

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    def _handler(path) -> logging.FileHandler:
        h = logging.FileHandler(path, encoding="utf-8")
        h.setFormatter(formatter)
        return h

    # Root "ragent" logger collects everything via child propagation
    root = logging.getLogger("ragent")
    root.setLevel(logging.DEBUG)
    root.addHandler(_handler(LOG_FILE))

    # Per-component loggers each write to their own file
    for name, log_file in [
        ("ragent.server", RAGENT_SERVER_LOG_FILE),
        ("ragent.vectordb", VECTORDB_LOG_FILE),
        ("ragent.llm", LLM_LOG_FILE),
    ]:
        logging.getLogger(name).addHandler(_handler(log_file))
