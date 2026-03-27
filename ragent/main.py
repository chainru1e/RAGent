"""Main dispatcher: reads stdin JSON and routes to the appropriate handler."""

import json
import logging
import sys

from ragent.config import LOG_FILE, ensure_dirs


def _setup_logging() -> None:
    """Configure file-based logging."""
    ensure_dirs()
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run() -> None:
    """Entry point: read stdin, dispatch to handler, never crash."""
    _setup_logging()
    logger = logging.getLogger("ragent")

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            logger.debug("Empty stdin, exiting")
            sys.exit(0)

        data = json.loads(raw)
        event = data.get("hook_event_name", "")
        logger.info("Received event: %s", event)

        if event == "UserPromptSubmit":
            from ragent.handlers.user_prompt_submit import handle
            handle(data)
        elif event == "Stop":
            from ragent.handlers.stop import handle
            handle(data)
        elif event == "SessionEnd":
            from ragent.handlers.session_end import handle
            handle(data)
        else:
            logger.warning("Unknown event: %s", event)

    except Exception:
        logger.exception("Unhandled error in ragent")

    # Always exit 0 to never disrupt Claude Code
    sys.exit(0)
