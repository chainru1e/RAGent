"""Codex CLI 어댑터.

Codex CLI 의 hook 이벤트(UserPromptSubmit/Stop 등) 와 rollout JSONL
transcript 를 RAGent 의 정규화 인터페이스로 매핑한다.

이벤트 매핑:
- UserPromptSubmit → "prompt"   (Claude Code 와 동일 이름)
- Stop             → "response"  (Claude Code 와 동일 이름)

Hook 입력 JSON 키 이름은 Claude Code 와 호환 (session_id / transcript_path /
hook_event_name / prompt / stop_hook_active 등) 이라 기존 handler 에 그대로
위임 가능 — 어댑터 내부 형식 변환 불필요.

Transcript 파싱은 CodexParser 가 담당. RolloutLine envelope ({type, payload})
를 ClaudeCodeParser 와 같은 prefix tag 컨벤션의 NormalizedMessage 로 정규화하므로
chunker / handler 변경 없이 동작한다.

설치 위치: ~/.codex/hooks.json (JSON 형식). config.toml 의 inline [hooks] 도
공식적으로 지원되나 TOML 작성에 필요한 외부 의존성을 회피하기 위해 본 phase
는 hooks.json 만 기본 등록 위치로 사용한다.
"""

import json
import sys
from pathlib import Path

from ragent.adapters.base import BaseAdapter
from ragent.modules.parsing_modules import CodexParser

_HOOKS_PATH = Path.home() / ".codex" / "hooks.json"
_RAGENT_DIR = Path(__file__).resolve().parent.parent.parent


class CodexAdapter(BaseAdapter):
    """Codex CLI hook 입력을 정규화 이벤트로 매핑하고 handler 에 위임한다."""

    parser_class = CodexParser

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
        """Codex 의 user-level hook 설정에 RAGent 를 등록한다.

        등록 위치: ~/.codex/hooks.json. config.toml 도 공식 지원되나 TOML
        write 의존성을 회피하기 위해 JSON 만 사용. 기존에 ~/.codex/config.toml
        의 [hooks] 테이블에 RAGent 가 등록돼 있다면 본 install 은 그것을
        건드리지 않으므로 우선순위 충돌 가능성 있음 — Codex CLI 가 두 소스를
        어떻게 머지하는지는 공식 문서가 명시 답하지 않음.

        등록 이벤트: UserPromptSubmit, Stop.

        멱등성: command 문자열에 "-m ragent" 가 포함된 기존 entry 를 모두
        제거한 뒤 새 hook 을 append. 두 번 호출해도 중복 등록되지 않는다.
        """
        # NOTE: VAR=value command prefix 는 POSIX 쉘에서만 동작. Codex 가
        # hook command 를 shell 경유로 실행하는지 공식 문서가 명시하지 않으나
        # 공식 예시에 $(...) 와 ~ 가 등장하므로 shell layer 를 거친다고 추정.
        # 이 가정이 깨지면 hook entry 의 env 필드 또는 wrapper script 방식으로
        # 마이그레이션 필요.
        cmd = (
            f"PYTHONPATH={_RAGENT_DIR} RAGENT_ADAPTER=codex "
            f"{sys.executable} -m ragent"
        )
        hooks_config = {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": cmd, "timeout": 5}]}
            ],
            "Stop": [
                {"hooks": [{"type": "command", "command": cmd, "timeout": 600}]}
            ],
        }

        settings: dict = {}
        if _HOOKS_PATH.exists():
            try:
                settings = json.loads(_HOOKS_PATH.read_text())
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {_HOOKS_PATH}, starting fresh")

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

        _HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HOOKS_PATH.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"RAGent hooks installed to {_HOOKS_PATH}")
        print(f"Command: {cmd}")
