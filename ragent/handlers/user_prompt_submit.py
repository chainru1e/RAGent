"""Handler for UserPromptSubmit hook event.

Saves the user prompt to a pending file for later pairing with the assistant response.
Must be fast (<50ms) - no DB access.
"""

import logging
from datetime import datetime, timezone

from ragent.pending import save_pending_prompt

logger = logging.getLogger("ragent")


def handle(data: dict) -> None:
    """Save the user's prompt to a pending file."""
    session_id = data.get("session_id", "")
    prompt = data.get("prompt", "")

    if not session_id or not prompt:
        logger.warning("UserPromptSubmit: missing session_id or prompt")
        return

    timestamp = datetime.now(timezone.utc).isoformat()
    save_pending_prompt(session_id, prompt, timestamp)
    logger.debug("UserPromptSubmit: saved pending prompt for session %s", session_id)
