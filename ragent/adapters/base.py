"""에이전트 어댑터 추상 베이스.

각 어댑터는 자신이 호스팅되는 에이전트(Claude Code, Codex 등)의 입력 스키마를
정규화된 이벤트 종류('prompt', 'response', 'session_end', 'unknown')로 번역한다.
정규화 이후의 라우팅은 BaseAdapter.dispatch가 모든 어댑터에 대해 동일하게 처리한다.
"""

import logging
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """모든 에이전트 어댑터의 추상 베이스."""

    @abstractmethod
    def on_prompt(self, data: dict) -> None:
        """사용자 프롬프트가 들어올 때 호출. 검색 결과를 stdout JSON으로 출력하는 책임은 구현체에 있다."""

    @abstractmethod
    def on_response(self, data: dict) -> None:
        """에이전트 응답이 완료될 때 호출. 인덱싱 트리거."""

    @abstractmethod
    def on_session_end(self, data: dict) -> None:
        """세션 종료 시 호출. transcript 전체를 다시 파싱하여 누락된 인덱싱을 보충."""

    @abstractmethod
    def _resolve_event_kind(self, data: dict) -> str:
        """입력 데이터를 정규화된 이벤트 종류 문자열로 변환한다.

        반환값은 다음 중 하나여야 한다: 'prompt', 'response', 'session_end', 'unknown'.
        에이전트 고유의 hook 이름·스키마 해석은 이 메서드 안에서만 한다.
        """

    def dispatch(self, data: dict) -> None:
        """모든 어댑터 공통 라우팅. 오버라이드 금지(원칙적으로).

        어댑터별 dispatch 변경이 필요해 보이면 _resolve_event_kind 매핑 한계가 아니라
        아키텍처 재논의 신호다.
        """
        kind = self._resolve_event_kind(data)
        if kind == "prompt":
            self.on_prompt(data)
        elif kind == "response":
            self.on_response(data)
        elif kind == "session_end":
            self.on_session_end(data)
        elif kind == "unknown":
            logger = logging.getLogger("ragent")
            logger.warning(
                "Adapter %s could not resolve event from data",
                type(self).__name__,
            )
        else:
            raise ValueError(
                f"_resolve_event_kind returned invalid value: {kind!r}. "
                f"Must be one of: prompt, response, session_end, unknown."
            )
