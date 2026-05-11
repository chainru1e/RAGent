"""에이전트 어댑터 추상 베이스.

각 어댑터는 자신이 호스팅되는 에이전트(Claude Code, Codex 등)의 입력 스키마를
정규화된 이벤트 종류('prompt', 'response', 'session_end', 'unknown')로 번역한다.
정규화 이후의 라우팅은 BaseAdapter.dispatch가 모든 어댑터에 대해 동일하게 처리한다.
"""

import logging
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """모든 에이전트 어댑터의 추상 베이스."""

    parser_class: type | None = None
    """이 어댑터의 transcript 를 처리할 BaseParser 서브클래스.

    어댑터는 hook 책임만 가지고 transcript 파싱 책임은 parser_class 에
    위임한다. 파싱 불가능한 어댑터(예: 미구현 stub) 는 None.

    타입을 type[BaseParser] 로 좁히지 않는 이유는 BaseAdapter 가 BaseParser 를
    직접 import 하면 순환 의존이 생길 수 있기 때문 — 각 어댑터 서브클래스가
    자기 parser 모듈을 import 해서 등록한다.
    """

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

    @classmethod
    @abstractmethod
    def install(cls) -> None:
        """이 어댑터에 해당하는 에이전트의 hook 시스템에 RAGent 호출을 등록한다.

        각 어댑터는 자신의 에이전트가 사용하는 hook 설정 파일 위치, 스키마,
        이벤트 이름을 알고 처리한다. 등록되는 hook 명령에는 환경변수
        RAGENT_ADAPTER가 명시적으로 박혀 있어야 한다 — RAGent 시작 시점에
        어떤 어댑터를 쓸지 결정하는 신호이기 때문이다.

        멱등해야 한다 — 두 번 호출해도 hook이 중복 등록되지 않아야 한다.
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
