"""Handler for SessionEnd hook event.

Parses the full transcript, upserts all Q&A pairs, generates a session summary,
and indexes it in ChromaDB.
"""

import logging

logger = logging.getLogger("ragent")


def handle(data: dict) -> None:
    pass
