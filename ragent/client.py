"""HTTP client helpers for the local RAGent server."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib import error, request

from ragent.config import RAGENT_SERVER_URL

logger = logging.getLogger("ragent")


def post_json(path: str, payload: dict[str, Any], timeout: float = 4.0) -> dict[str, Any] | None:
    """POST JSON to the local RAGent server.

    Hook handlers must never break the host agent, so connection and response
    failures are logged and converted to None.
    """
    url = f"{RAGENT_SERVER_URL.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except (OSError, error.URLError, TimeoutError):
        logger.exception("Failed to call RAGent server: %s", url)
        return None

    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        logger.warning("RAGent server returned invalid JSON from %s", url)
        return None

    if not isinstance(data, dict):
        logger.warning("RAGent server returned non-object JSON from %s", url)
        return None
    return data
