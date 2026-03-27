"""JSONL transcript parser for Claude Code conversations."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Turn:
    """A single user-assistant exchange."""
    user_prompt: str
    assistant_response: str
    user_uuid: str = ""
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)


# Patterns for content to skip
_SKIP_PATTERNS = [
    re.compile(r"<command-name>"),
    re.compile(r"<local-command-"),
]


def _should_skip_content(text: str) -> bool:
    """Check if content matches skip patterns."""
    return any(p.search(text) for p in _SKIP_PATTERNS)


def _extract_user_prompt(entry: dict) -> str | None:
    """Extract user prompt text from a transcript entry.

    Returns None if the entry should be skipped.
    """
    if entry.get("isMeta"):
        return None

    message = entry.get("message", {})
    content = message.get("content")

    # Only use entries where content is a plain string (not tool_result lists)
    if not isinstance(content, str):
        return None

    if _should_skip_content(content):
        return None

    return content.strip() if content.strip() else None


def _extract_assistant_text(entry: dict) -> str:
    """Extract text blocks from an assistant message, skipping thinking/tool_use."""
    message = entry.get("message", {})
    content = message.get("content", [])

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    texts.append(text)
        return "\n\n".join(texts)

    return ""


def parse_transcript(path: str | Path) -> list[Turn]:
    """Parse a full JSONL transcript into a list of Turns."""
    path = Path(path)
    if not path.exists():
        return []

    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    turns = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        role = entry.get("type") or entry.get("message", {}).get("role")

        if role in ("user", "human"):
            prompt = _extract_user_prompt(entry)
            if prompt:
                # Look for the next assistant response
                assistant_text = ""
                j = i + 1
                while j < len(entries):
                    next_entry = entries[j]
                    next_role = next_entry.get("type") or next_entry.get("message", {}).get("role")
                    if next_role in ("assistant",):
                        assistant_text = _extract_assistant_text(next_entry)
                        if assistant_text:
                            break
                    elif next_role in ("user", "human"):
                        break
                    j += 1

                turn = Turn(
                    user_prompt=prompt,
                    assistant_response=assistant_text,
                    user_uuid=entry.get("uuid", ""),
                    timestamp=entry.get("timestamp", ""),
                )
                turns.append(turn)
        i += 1

    return turns


def get_last_turn(path: str | Path) -> Turn | None:
    """Get only the last user-assistant turn from a transcript."""
    turns = parse_transcript(path)
    return turns[-1] if turns else None


def get_session_summary_text(turns: list[Turn]) -> str:
    """Generate a session summary text from a list of turns."""
    if not turns:
        return ""

    lines = []
    for i, turn in enumerate(turns, 1):
        q = turn.user_prompt[:200]
        a = turn.assistant_response[:300]
        lines.append(f"Turn {i}:\nQ: {q}\nA: {a}")

    return "\n\n".join(lines)
