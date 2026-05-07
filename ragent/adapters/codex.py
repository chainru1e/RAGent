"""Codex 어댑터 (현재 stub).

RAGent 어댑터 패턴의 두 번째 사례로 ADAPTER_REGISTRY와 BaseAdapter
추상화를 검증하기 위해 추가됨. Codex hook 시스템 명세가 조사되면
이 stub을 ClaudeCodeAdapter와 같은 깊이로 채울 예정.

활용 시점:
- 환경변수 RAGENT_ADAPTER=codex로 RAGent를 부르면 dispatch까지는 도달함
- 단 _resolve_event_kind가 "unknown"만 반환하므로 어떤 hook도
  실제 처리는 일어나지 않음 (경고 로그 후 종료)
"""

from ragent.adapters.base import BaseAdapter


class CodexAdapter(BaseAdapter):
    """Codex hook 입력을 정규화 이벤트로 매핑할 예정인 stub 어댑터."""

    def _resolve_event_kind(self, data: dict) -> str:
        # TODO: Codex hook 시스템 명세 미조사. 명세 확정 후 매핑 표 작성:
        # - Codex의 user prompt 도착 이벤트 이름 → "prompt"
        # - Codex의 응답 완료 이벤트 이름 → "response"
        # - Codex의 세션 종료 이벤트 이름 → "session_end"
        return "unknown"

    def on_prompt(self, data: dict) -> None:
        raise NotImplementedError(
            "CodexAdapter.on_prompt not yet implemented. See <Codex hook docs>"
        )

    def on_response(self, data: dict) -> None:
        raise NotImplementedError(
            "CodexAdapter.on_response not yet implemented. See <Codex hook docs>"
        )

    def on_session_end(self, data: dict) -> None:
        raise NotImplementedError(
            "CodexAdapter.on_session_end not yet implemented. See <Codex hook docs>"
        )

    @classmethod
    def install(cls) -> None:
        raise NotImplementedError(
            "Codex hook installation not yet implemented. "
            "Required: hooks.json schema research, install path discovery."
        )
