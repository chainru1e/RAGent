"""Claude Code 어댑터.

Claude Code의 hook 이벤트 이름(UserPromptSubmit/Stop 등)은
이 모듈 안에서만 다룬다. 무거운 의존성(qdrant, FlagEmbedding 등) 로드를
hook 호출 시점까지 미루기 위해 handler import는 lazy로 수행한다.
"""

import json
from pathlib import Path

from ragent.adapters.base import BaseAdapter
from ragent.modules.parsing_modules import ClaudeCodeParser

_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

# get_adapter(name) 가 ADAPTER_REGISTRY 에서 이 어댑터를 찾는 키.
_ADAPTER_NAME = "claude_code"


class ClaudeCodeAdapter(BaseAdapter):
    """Claude Code hook 입력을 정규화 이벤트로 매핑하고 handler에 위임한다."""

    parser_class = ClaudeCodeParser

    def _resolve_event_kind(self, data: dict) -> str:
        event = data.get("hook_event_name", "")
        return {
            "UserPromptSubmit": "prompt",
            "Stop": "response",
        }.get(event, "unknown")

    def on_prompt(self, data: dict) -> None:
        from ragent.handlers.user_prompt_submit import handle
        handle(data)

    def on_response(self, data: dict) -> None:
        from ragent.handlers.stop import handle
        handle(data)

    @classmethod
    def install(cls) -> None:
        """Claude Code의 user-level settings.json에 RAGent hook을 등록한다.

        등록 위치: ~/.claude/settings.json (user-level only). project-level
        (./.claude/settings.json)은 지원하지 않는다.

        등록되는 hook 명령에는 어댑터 선택자가 `--adapter claude_code` 인자로
        박힌다. hook 진입점(main.run)이 이 인자를 읽어 get_adapter 에 넘긴다.
        환경변수 프리픽스(VAR=value)를 쓰지 않으므로 Windows(cmd)/POSIX(sh)
        양쪽에서 동작한다. frozen(exe) 모드는 install exe 와 같은 디렉터리의
        hook exe 를, 소스 모드는 활성 파이썬의 `-m ragent` 를 가리킨다.

        멱등성: RAGent hook 으로 식별되는(is_ragent_hook) 기존 entry 를 모두
        제거한 뒤 새 hook 을 append 한다. 따라서 두 번 호출해도 중복 등록되지
        않는다.
        """
        # 셸 비종속 hook 명령(frozen exe 경로 또는 `-m ragent`)을 BaseAdapter 가
        # 생성한다. 어댑터 선택자는 --adapter 인자로 전달된다.
        cmd = cls.build_hook_command(_ADAPTER_NAME)
        hooks_config = {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": cmd, "timeout": 5}]}
            ],
            "Stop": [
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
                    cls.is_ragent_hook(h.get("command", ""))
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
