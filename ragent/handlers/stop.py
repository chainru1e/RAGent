"""Handler for Stop hook event.

Pairs the pending prompt with the assistant's response from the transcript
and indexes the Q&A pair in ChromaDB.
"""

import logging
import os

from ragent.modules.chunking_modules import *
from ragent.modules.intent_classifying_modules import *
from ragent.modules.embedding_modules import *
from ragent.vectordb import *
from ragent.config import GEMINI_API_KEY

logger = logging.getLogger("ragent")


def handle(data: dict) -> None:
    """Index the last Q&A pair from the conversation."""
    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path", "")
    stop_hook_active = data.get("stop_hook_active", False)

    if stop_hook_active:
        logger.debug("Stop: stop_hook_active is True, skipping to prevent loop")
        return

    if not session_id:
        logger.warning("Stop: missing session_id")
        return

    if not transcript_path:
        logger.warning("Stop: missing transcript_path")
        return

    from ragent.adapters import get_adapter
    adapter_cls = type(get_adapter(None))
    if adapter_cls.parser_class is None:
        logger.warning(
            "Stop: %s has no parser_class, skipping", adapter_cls.__name__
        )
        return

    parser = adapter_cls.parser_class(transcript_path)
    chunker = Chunker()
    intent_classifier = IntentClassifier()
    embedder = HybridEmbedding()
    vectordb = QdrantStorage(os.path.basename(os.path.dirname(transcript_path)))

    last_turn = parser.parse_last_turn()

    if not last_turn:
        logger.warning("Stop: no turns found in transcript %s", transcript_path)
        return

    try:
        count = index_turn(
            last_turn,
            session_id=session_id,
            chunker=chunker,
            intent_classifier=intent_classifier,
            embedder=embedder,
            vectordb=vectordb,
        )
        logger.info("Stop: indexed %d chunks for session %s", count, session_id)
    finally:
        vectordb.close()