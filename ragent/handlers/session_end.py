"""Handler for SessionEnd hook event.

Parses the entire transcript JSONL via the active adapter's parser_class and
upserts every genuine user/assistant turn to Qdrant. Acts as full recovery for
any turns that the Stop handler missed. Idempotent within a process: chunk_ids
are overridden with deterministic values derived from session_id + turn index
so re-running on the same transcript hits Qdrant upsert without duplicating
points.
"""

import logging
import os

from ragent.modules.chunking_modules import Chunker
from ragent.modules.intent_classifying_modules import HybridClassifier
from ragent.modules.embedding_modules import HybridEmbedding
from ragent.vectordb import QdrantStorage
from ragent.config import GEMINI_API_KEY

logger = logging.getLogger("ragent")


def handle(data: dict) -> None:
    """SessionEnd hook handler.

    Parses the entire transcript JSONL and upserts all genuine user/assistant
    turns to the vectordb. Idempotent — re-running on the same transcript
    must not duplicate chunks (relies on Qdrant upsert by chunk_id)."""
    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path", "")

    if not session_id:
        logger.warning("SessionEnd: missing session_id")
        return

    if not transcript_path:
        logger.warning("SessionEnd: missing transcript_path")
        return

    if not os.path.exists(transcript_path):
        logger.warning("SessionEnd: transcript not found at %s", transcript_path)
        return

    from ragent.adapters import get_adapter
    adapter_cls = type(get_adapter(None))
    if adapter_cls.parser_class is None:
        logger.warning(
            "SessionEnd: %s has no parser_class, skipping", adapter_cls.__name__
        )
        return

    parser = adapter_cls.parser_class(transcript_path)
    turns = parser.parse_full_transcript()

    if not turns:
        logger.warning("SessionEnd: no parsable messages in %s", transcript_path)
        return

    total_messages = sum(len(t) for t in turns)
    logger.info(
        "SessionEnd: parsed %d messages into %d turns from %s",
        total_messages, len(turns), transcript_path,
    )

    chunker = Chunker()
    intent_classifier = HybridClassifier(GEMINI_API_KEY)
    embedder = HybridEmbedding()
    vectordb = QdrantStorage(os.path.basename(os.path.dirname(transcript_path)))

    try:
        all_chunks = []
        for turn_idx, turn in enumerate(turns):
            chunks = chunker.process_turn(turn)
            if not chunks:
                continue
            # Chunker가 부여한 UUID 기반 chunk_id를 결정적 ID로 덮어써,
            # 같은 transcript 재처리 시 같은 point ID로 매핑되도록 한다.
            parent_id = f"{session_id}_turn_{turn_idx}"
            chunks[0].metadata.chunk_id = parent_id
            for code_idx, c in enumerate(chunks[1:]):
                c.metadata.parent_id = parent_id
                c.metadata.chunk_id = f"{parent_id}_code_{code_idx}"

            intent = intent_classifier.classify(chunks[0].payload).category
            for c in chunks:
                c.metadata.type = intent
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("SessionEnd: no chunks produced from %s", transcript_path)
            return

        texts = [c.payload for c in all_chunks]
        vectors = embedder.embed_batch(texts, batch_size=32)
        for c, v in zip(all_chunks, vectors):
            c.vector = v

        count = vectordb.add_points_batch(all_chunks)
        logger.info(
            "SessionEnd: indexed %d chunks across %d turns for session %s",
            count, len(turns), session_id,
        )
    finally:
        vectordb.close()
