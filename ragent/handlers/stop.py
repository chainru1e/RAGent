"""Handler for Stop hook event."""

import logging

from ragent.client import post_json

logger = logging.getLogger("ragent")


def handle(data: dict) -> None:
    """Ask the local RAGent server to index the last conversation turn."""
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

    response = post_json(
        "/save",
        {
            "session_id": session_id,
            "transcript_path": transcript_path,
        },
        timeout=5.0,
    )
    if response is None:
        return

    if not response.get("ok", False):
        logger.warning("Stop: server rejected save request: %s", response)
