"""Install RAGent hooks into Claude Code settings."""

import json
import sys
from pathlib import Path

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
RAGENT_DIR = Path(__file__).resolve().parent


def get_hook_command() -> str:
    """Build the hook command with PYTHONPATH set."""
    return f"PYTHONPATH={RAGENT_DIR} {sys.executable} -m ragent"


def build_hooks_config() -> dict:
    """Build the hooks configuration to merge into settings."""
    cmd = get_hook_command()
    return {
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": cmd,
                        "timeout": 5,
                    }
                ]
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": cmd,
                        "timeout": 600,
                    }
                ]
            }
        ],
        "SessionEnd": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": cmd,
                        "timeout": 600,
                    }
                ]
            }
        ],
    }


def install() -> None:
    """Merge RAGent hooks into Claude Code settings.json."""
    # Read existing settings
    settings: dict = {}
    if SETTINGS_PATH.exists():
        try:
            settings = json.loads(SETTINGS_PATH.read_text())
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {SETTINGS_PATH}, starting fresh")

    # Merge hooks
    existing_hooks = settings.get("hooks", {})
    ragent_hooks = build_hooks_config()

    for event_name, hook_entries in ragent_hooks.items():
        if event_name not in existing_hooks:
            existing_hooks[event_name] = []

        # Remove any existing RAGent hooks to avoid duplicates
        existing_hooks[event_name] = [
            entry
            for entry in existing_hooks[event_name]
            if not any(
                "ragent" in h.get("command", "")
                for h in entry.get("hooks", [])
            )
        ]

        # Add RAGent hooks
        existing_hooks[event_name].extend(hook_entries)

    settings["hooks"] = existing_hooks

    # Write back
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    print(f"RAGent hooks installed to {SETTINGS_PATH}")
    print(f"Command: {get_hook_command()}")


if __name__ == "__main__":
    install()
