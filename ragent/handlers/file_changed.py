"""Handler for successful file-changing tool events."""

import logging
import os
import re
from typing import Any

from ragent.client import post_json

logger = logging.getLogger("ragent")


_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$")


def handle(data: dict) -> None:
    """Queue whole-file snapshot indexing for changed files."""
    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path", "")
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input") or data.get("input") or {}
    workspace_root = (
        data.get("cwd")
        or data.get("workspace_root")
        or data.get("workspace_dir")
        or os.getcwd()
    )

    if not session_id:
        logger.warning("FileChanged: missing session_id")
        return

    paths = _extract_changed_paths(tool_name, tool_input)
    if not paths:
        logger.debug("FileChanged: no changed paths found for tool %s", tool_name)
        return

    response = post_json(
        "/file_changed",
        {
            "session_id": session_id,
            "transcript_path": transcript_path,
            "workspace_root": workspace_root,
            "paths": sorted(set(paths)),
        },
        timeout=5.0,
    )
    if response is None:
        return

    if not response.get("ok", False):
        logger.warning("FileChanged: server rejected request: %s", response)


def _extract_changed_paths(tool_name: str, tool_input: Any) -> list[str]:
    if not isinstance(tool_input, dict):
        return []

    if tool_name in {"Edit", "Write", "MultiEdit"}:
        file_path = tool_input.get("file_path")
        return [file_path] if isinstance(file_path, str) and file_path else []

    if tool_name == "apply_patch":
        patch_text = _stringify_patch_input(tool_input)
        return _extract_paths_from_patch(patch_text)

    return []


def _stringify_patch_input(tool_input: dict) -> str:
    for key in ("command", "cmd", "patch", "input"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return "\n".join(value for value in tool_input.values() if isinstance(value, str))


def _extract_paths_from_patch(patch_text: str) -> list[str]:
    paths: list[str] = []
    for raw_line in patch_text.splitlines():
        line = raw_line.strip()
        match = _PATCH_FILE_RE.match(line)
        if match:
            paths.append(match.group(1).strip())
    return paths
