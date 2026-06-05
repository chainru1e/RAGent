"""Main dispatcher: reads stdin JSON and routes to the appropriate handler."""

import argparse
import json
import logging
import sys

from ragent.logging_config import setup_logging


def run() -> None:
    """Entry point: read stdin, dispatch to handler, never crash."""
    setup_logging()
    logger = logging.getLogger("ragent")

    # 어댑터 선택자는 hook 커맨드에 --adapter 인자로 박혀 들어온다(cross-platform).
    # 미지정 시 get_adapter 가 RAGENT_ADAPTER 환경변수 → 기본값 순으로 폴백한다.
    parser = argparse.ArgumentParser(prog="ragent", add_help=False)
    parser.add_argument("--adapter", default=None)
    args, _ = parser.parse_known_args()

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            logger.debug("Empty stdin, exiting")
            sys.exit(0)

        data = json.loads(raw)
        event = data.get("hook_event_name", "")
        logger.info("Received event: %s", event)

        from ragent.adapters import get_adapter
        adapter = get_adapter(args.adapter)
        adapter.dispatch(data)

    except Exception:
        logger.exception("Unhandled error in ragent")

    # Always exit 0 to never disrupt Claude Code
    sys.exit(0)
