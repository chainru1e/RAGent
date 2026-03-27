"""Pending prompt file management with atomic writes."""

import json
import os
import tempfile
from pathlib import Path

from ragent.config import PENDING_DIR, ensure_dirs


def save_pending_prompt(session_id: str, prompt: str, timestamp: str) -> None:
    """Save a pending prompt atomically (.tmp -> rename)."""
    ensure_dirs()
    data = {
        "session_id": session_id,
        "prompt": prompt,
        "timestamp": timestamp,
    }
    target = PENDING_DIR / f"{session_id}.json"
    fd, tmp_path = tempfile.mkstemp(dir=PENDING_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.rename(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_pending_prompt(session_id: str) -> dict | None:
    """Load a pending prompt, or return None if not found."""
    target = PENDING_DIR / f"{session_id}.json"
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def delete_pending_prompt(session_id: str) -> None:
    """Delete a pending prompt file if it exists."""
    target = PENDING_DIR / f"{session_id}.json"
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass
