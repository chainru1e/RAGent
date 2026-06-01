"""Windsurf (Cognition Cascade) 어댑터.

Cascade Hooks 의 hook 이벤트와 transcript JSONL 을 RAGent 의 정규화 인터페이스로
매핑한다. 공식 출처: https://docs.windsurf.com/windsurf/cascade/hooks.

이벤트 매핑:
- pre_user_prompt                          → "prompt"
- post_cascade_response_with_transcript    → "response"

Windsurf 는 SessionEnd 대응 이벤트가 없다. RAGent 자체도 SessionEnd 처리를
커밋 44e5461 / 8806cff 에서 제거한 상태라 이 부재가 어댑터 간 비대칭이 아니라
공통 분모로 정착됐다.

키 정규화: Windsurf hook payload 는 Claude Code / Codex 와 키 이름이 다르므로
어댑터 안에서 다음 매핑으로 변환해 handler 에 dict 그대로 위임한다:
- trajectory_id                → session_id
- tool_info.transcript_path    → transcript_path  (post_cascade_response_with_transcript 만 제공)
- tool_info.user_prompt        → prompt           (pre_user_prompt 만 제공)
- agent_action_name            → hook_event_name

pre_user_prompt 시점에는 hook payload 에 transcript_path 가 들어오지 않는다.
공식 docs 가 명시한 표준 위치 ~/.windsurf/transcripts/{trajectory_id}.jsonl 로
합성한다 — user_prompt_submit handler 는 transcript 본문을 읽지 않고 dirname
만으로 vectordb collection 이름을 추출하므로 해당 시점에 파일이 존재하지 않아도
영향 없음. dirname 이 모든 Windsurf 세션에 대해 동일한 ~/.windsurf/transcripts
가 되어 collection 이 단일로 묶인다는 한계는 별도 phase 에서 (workspace 경로를
hook payload 에서 얻을 수 있는 시점부터) 해소.

Transcript 파싱은 WindsurfParser 가 담당. 공식 docs 가 공개한 3개 step type
(user_input / planner_response / code_action) 만 정규화하고, 미공개 9개 type
은 모두 None — 추측 매핑 금지.

chunker 호환성: code_action 의 prefix tag 는 "[code_action]" 으로 부여한다.
Claude Code 의 "[Write]" 와 비호환이라 코드 청크 인덱싱은 발화하지 않음 —
Codex 의 apply_patch 와 동일 패턴.

설치 위치: ~/.codeium/windsurf/hooks.json (user-level). system-level 과
workspace-level (.windsurf/hooks.json) 은 본 phase 에서 다루지 않음.
"""

import json
import sys
from pathlib import Path

from ragent.adapters.base import BaseAdapter
from ragent.modules.parsing_modules import WindsurfParser

_HOOKS_PATH = Path.home() / ".codeium" / "windsurf" / "hooks.json"
_TRANSCRIPTS_DIR = Path.home() / ".windsurf" / "transcripts"
_RAGENT_DIR = Path(__file__).resolve().parent.parent.parent


class WindsurfAdapter(BaseAdapter):
    """Windsurf (Cognition Cascade) hook 입력을 정규화 이벤트로 매핑한다."""

    parser_class = WindsurfParser

    def _resolve_event_kind(self, data: dict) -> str:
        event = data.get("agent_action_name", "")
        return {
            "pre_user_prompt": "prompt",
            "post_cascade_response_with_transcript": "response",
        }.get(event, "unknown")

    def on_prompt(self, data: dict) -> None:
        from ragent.handlers.user_prompt_submit import handle
        handle(self._normalize_windsurf_payload(data))

    def on_response(self, data: dict) -> None:
        from ragent.handlers.stop import handle
        handle(self._normalize_windsurf_payload(data))

    @staticmethod
    def _normalize_windsurf_payload(data: dict) -> dict:
        # post_cascade_response_with_transcript 는 tool_info.transcript_path 로
        # 절대 경로 제공. pre_user_prompt 는 transcript_path 미제공이라 공식
        # 표준 위치로 합성.
        tool_info = data.get("tool_info") or {}
        trajectory_id = data.get("trajectory_id", "")
        transcript_path = tool_info.get("transcript_path") or (
            str(_TRANSCRIPTS_DIR / f"{trajectory_id}.jsonl")
            if trajectory_id
            else ""
        )
        return {
            "session_id": trajectory_id,
            "transcript_path": transcript_path,
            "prompt": tool_info.get("user_prompt", ""),
            "hook_event_name": data.get("agent_action_name", ""),
        }

    @classmethod
    def install(cls) -> None:
        """~/.codeium/windsurf/hooks.json 에 RAGent hook 을 등록한다.

        등록 이벤트: pre_user_prompt, post_cascade_response_with_transcript.
        SessionEnd 대응 이벤트는 Windsurf 에 없으며 RAGent 도 SessionEnd 처리를
        이미 제거한 상태라 등록 대상 없음.

        Windsurf hooks.json 스키마는 Claude Code / Codex 와 달리 hook entry 가
        `{"command": "...", "powershell": "...", ...}` 형태로 한 단계 단순하다.
        Claude Code 의 `{"hooks": [{"type": "command", ...}]}` 추가 wrap 없음.

        멱등성: command 문자열에 "-m ragent" 가 포함된 기존 entry 를 모두
        제거한 뒤 새 hook 을 append. 두 번 호출해도 중복 등록되지 않는다.
        """
        # NOTE: VAR=value command prefix 는 POSIX 쉘에서만 동작. Windsurf 는
        # macOS/Linux 에서 command 필드를 bash -c 로 spawn 한다고 공식 docs 가
        # 명시하므로 inline prefix 가 자연 작동한다. Windows 의 powershell 필드는
        # 본 phase 에서 채우지 않음.
        cmd = (
            f"PYTHONPATH={_RAGENT_DIR} RAGENT_ADAPTER=windsurf "
            f"{sys.executable} -m ragent"
        )
        new_entries = {
            "pre_user_prompt": [{"command": cmd}],
            "post_cascade_response_with_transcript": [{"command": cmd}],
        }

        settings: dict = {}
        if _HOOKS_PATH.exists():
            try:
                settings = json.loads(_HOOKS_PATH.read_text())
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {_HOOKS_PATH}, starting fresh")

        existing_hooks = settings.get("hooks", {})
        for event_name, entries in new_entries.items():
            current = existing_hooks.get(event_name, [])
            current = [
                entry
                for entry in current
                if "-m ragent" not in entry.get("command", "")
            ]
            current.extend(entries)
            existing_hooks[event_name] = current

        settings["hooks"] = existing_hooks

        _HOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HOOKS_PATH.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"RAGent hooks installed to {_HOOKS_PATH}")
        print(f"Command: {cmd}")
