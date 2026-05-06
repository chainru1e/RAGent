"""Claude Code 어댑터.

Claude Code의 hook 이벤트 이름(UserPromptSubmit/Stop/SessionEnd 등)은
이 모듈 안에서만 다룬다. 무거운 의존성(qdrant, FlagEmbedding 등) 로드를
hook 호출 시점까지 미루기 위해 handler import는 lazy로 수행한다.
"""

from ragent.adapters.base import BaseAdapter


class ClaudeCodeAdapter(BaseAdapter):
    """Claude Code hook 입력을 정규화 이벤트로 매핑하고 handler에 위임한다."""

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
