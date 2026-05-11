"""Claude Code 어댑터.

Claude Code의 hook 이벤트 이름(UserPromptSubmit/Stop/SessionEnd 등)은
이 모듈 안에서만 다룬다. 무거운 의존성(qdrant, FlagEmbedding 등) 로드를
hook 호출 시점까지 미루기 위해 handler import는 lazy로 수행한다.
"""

import json
import sys
from pathlib import Path

from ragent.adapters.base import BaseAdapter
from ragent.modules.parsing_modules import ClaudeCodeParser

_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
_RAGENT_DIR = Path(__file__).resolve().parent.parent.parent


class ClaudeCodeAdapter(BaseAdapter):
    """Claude Code hook 입력을 정규화 이벤트로 매핑하고 handler에 위임한다."""

    parser_class = ClaudeCodeParser

    def _resolve_event_kind(self, data: dict) -> str:
        event = data.get("hook_event_name", "")
        return {
            "UserPromptSubmit": "prompt",
            "Stop": "response",
            "SessionEnd": "session_end",
        }.get(event, "unknown")

    def on_prompt(self, data: dict) -> None:
        from ragent.handlers.user_prompt_submit import handle
        handle(data)

    def on_response(self, data: dict) -> None:
        from ragent.handlers.stop import handle
        handle(data)

    def on_session_end(self, data: dict) -> None:
        from ragent.handlers.session_end import handle
        handle(data)

    @classmethod
    def install(cls) -> None:
        """Claude Code의 user-level settings.json에 RAGent hook을 등록한다.

        등록 위치: ~/.claude/settings.json (user-level only). project-level
        (./.claude/settings.json)은 지원하지 않는다.

        등록되는 hook 명령에는 RAGENT_ADAPTER=claude_code 환경변수가 inline
        prefix로 박힌다. RAGent 시작 시점에 ragent.adapters.get_adapter()가
        이 환경변수를 읽어 ClaudeCodeAdapter를 선택하기 때문이다.

        멱등성: command 문자열에 "-m ragent"가 포함된 기존 hook entry를 모두
        제거한 뒤 새 hook을 append한다. 따라서 두 번 호출해도 중복 등록되지
        않는다.
        """
        # NOTE: VAR=value command prefix는 POSIX 쉘에서만 동작.
        # Windows 지원 시점에는 settings.json hook entry의 env 필드 방식으로
        # 마이그레이션 검토 필요.
        cmd = (
            f"PYTHONPATH={_RAGENT_DIR} RAGENT_ADAPTER=claude_code "
            f"{sys.executable} -m ragent"
        )
        hooks_config = {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": cmd, "timeout": 5}]}
            ],
            "Stop": [
                {"hooks": [{"type": "command", "command": cmd, "timeout": 600}]}
            ],
            "SessionEnd": [
                {"hooks": [{"type": "command", "command": cmd, "timeout": 600}]}
            ],
        }

        settings: dict = {}
        if _SETTINGS_PATH.exists():
            try:
                settings = json.loads(_SETTINGS_PATH.read_text())
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {_SETTINGS_PATH}, starting fresh")

        existing_hooks = settings.get("hooks", {})
        for event_name, hook_entries in hooks_config.items():
            if event_name not in existing_hooks:
                existing_hooks[event_name] = []

            existing_hooks[event_name] = [
                entry
                for entry in existing_hooks[event_name]
                if not any(
                    "-m ragent" in h.get("command", "")
                    for h in entry.get("hooks", [])
                )
            ]

            existing_hooks[event_name].extend(hook_entries)

        settings["hooks"] = existing_hooks

        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_PATH.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"RAGent hooks installed to {_SETTINGS_PATH}")
        print(f"Command: {cmd}")
