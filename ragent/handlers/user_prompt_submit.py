"""Handler for UserPromptSubmit hook event."""

import json
import logging

from ragent.client import post_json

logger = logging.getLogger("ragent")


def handle(data: dict) -> None:
    """Request relevant context from the local RAGent server."""
    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path", "")
    prompt = data.get("prompt", "")

    if not session_id:
        logger.warning("UserPromptSubmit: missing session_id")
        return

    if not transcript_path:
        logger.warning("UserPromptSubmit: missing transcript_path")
        return

    if not prompt:
        logger.warning("UserPromptSubmit: missing prompt")
        return

    response = post_json(
        "/search",
        {
            "session_id": session_id,
            "transcript_path": transcript_path,
            "prompt": prompt,
        },
        timeout=30.0,
    )
    context = response.get("context", "") if response else ""

    if context:
        additional_context = f"Retrieved relevant code:\n{context}"
    else:
        additional_context = ""

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional_context,
        }
    }
    print(json.dumps(output, ensure_ascii=False))
