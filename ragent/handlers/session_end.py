"""Handler for SessionEnd hook event.

Parses the full transcript, upserts all Q&A pairs, generates a session summary,
and indexes it in ChromaDB.
"""

import logging

from ragent.pending import delete_pending_prompt
from ragent.conversation_collector import get_session_summary_text, parse_transcript
from ragent.vectordb import RAGentDB

logger = logging.getLogger("ragent")


def handle(data: dict) -> None:
    """Process the full session transcript and index everything."""
    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path", "")

    if not session_id or not transcript_path:
        logger.warning("SessionEnd: missing session_id or transcript_path")
        return

    turns = parse_transcript(transcript_path)
    if not turns:
        logger.info("SessionEnd: no turns found in transcript %s", transcript_path)
        return

    db = RAGentDB()

    # Upsert all Q&A pairs (covers anything Stop might have missed)
    indexed = 0
    for turn in turns:
        if turn["user_prompt"] and turn["assistant_response"]:
            db.index_qa_pair(turn)
            indexed += 1

    logger.info("SessionEnd: indexed %d Q&A pairs for session %s", indexed, session_id)

    # Generate and index session summary
    summary_text = get_session_summary_text(turns)
    if summary_text:
        metadata = {
            "turn_count": len(turns),
            "first_prompt": turns[0]["user_prompt"][:200],
        }
        db.index_session_summary(session_id, summary_text, metadata)
        logger.info("SessionEnd: indexed session summary for %s", session_id)

    # Clean up pending file
    delete_pending_prompt(session_id)
