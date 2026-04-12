"""Handler for UserPromptSubmit hook event.

Saves the user prompt to a pending file for later pairing with the assistant response.
Must be fast (<50ms) - no DB access.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("ragent")


def handle(data: dict) -> None:
    pass