"""Handler for Stop hook event.

Pairs the pending prompt with the assistant's response from the transcript
and indexes the Q&A pair in ChromaDB.
"""

import logging

from ragent.pending import delete_pending_prompt, load_pending_prompt
from ragent.transcript import Turn, get_last_turn
from ragent.vectordb import RAGentDB

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

    # Try pending prompt first, fall back to transcript
    pending = load_pending_prompt(session_id)
    last_turn = get_last_turn(transcript_path)

    if not last_turn:
        logger.warning("Stop: no turns found in transcript %s", transcript_path)
        return

    # Use pending prompt if available, otherwise use transcript's prompt
    if pending:
        turn = Turn(
            user_prompt=pending["prompt"],
            assistant_response=last_turn.assistant_response,
            user_uuid=last_turn.user_uuid,
            timestamp=pending.get("timestamp", last_turn.timestamp),
        )
    else:
        turn = last_turn

    if not turn.assistant_response:
        logger.debug("Stop: no assistant response found, skipping indexing")
        return

    db = RAGentDB()
    db.index_qa_pair(turn)
    logger.info("Stop: indexed Q&A pair for session %s", session_id)

    delete_pending_prompt(session_id)
