"""Handler for Stop hook event.

Pairs the pending prompt with the assistant's response from the transcript
and indexes the Q&A pair in ChromaDB.
"""

import logging
import os

from ragent.modules.parsing_modules import *
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

    parser = MessageParser(transcript_path)
    chunker = Chunker()
    intent_classifier = HybridClassifier(GEMINI_API_KEY)
    embedder = HybridEmbedding()
    vectordb = QdrantStorage(os.path.basename(os.path.dirname(transcript_path)))

    last_turn = parser.parse_last_turn()

    if not last_turn:
        logger.warning("Stop: no turns found in transcript %s", transcript_path)
        return

    chunks = chunker.process_turn(last_turn)

    context_chunk = next((chunk for chunk in chunks if chunk.metadata.chunk_id), None)
    if not context_chunk:
        logger.warning("Stop: no context chunk found")
        return

    intent = intent_classifier.classify(context_chunk.payload).category
    texts = [chunk.payload for chunk in chunks]
    vectors = embedder.embed_batch(texts, batch_size=32)

    for chunk, vector in zip(chunks, vectors):
        chunk.metadata.type = intent
        chunk.vector = vector

    try:
        count = vectordb.add_points_batch(chunks)
        logger.info("Stop: indexed %d chunks for session %s", count, session_id)
    finally:
        vectordb.close()