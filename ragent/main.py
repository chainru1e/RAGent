"""Main dispatcher: reads stdin JSON and routes to the appropriate handler."""

import json
import logging
import sys

from ragent.logging_config import setup_logging


def run() -> None:
    """Entry point: read stdin, dispatch to handler, never crash."""
    setup_logging()
    logger = logging.getLogger("ragent")

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            logger.debug("Empty stdin, exiting")
            sys.exit(0)

        data = json.loads(raw)
        event = data.get("hook_event_name", "")
        logger.info("Received event: %s", event)

        from ragent.adapters import get_adapter
        adapter = get_adapter(None)
        adapter.dispatch(data)

    except Exception:
        logger.exception("Unhandled error in ragent")

    # Always exit 0 to never disrupt Claude Code
    sys.exit(0)
