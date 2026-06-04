"""에이전트 어댑터 추상 베이스.

각 어댑터는 자신이 호스팅되는 에이전트(Claude Code, Codex 등)의 입력 스키마를
정규화된 이벤트 종류('prompt', 'response', 'unknown')로 번역한다.
정규화 이후의 라우팅은 BaseAdapter.dispatch가 모든 어댑터에 대해 동일하게 처리한다.
"""

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path

# frozen(개별 exe) 빌드 시 hook 진입점이 빌드되는 실행파일 이름(확장자 제외).
# launcher 가 기대하는 ragent-llm-server / ragent-server 와 동일한 명명 규약이며,
# .spec 산출물 이름이 이 값과 일치해야 install 이 올바른 hook 경로를 등록한다.
HOOK_EXE = "ragent-hook"


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
    def _resolve_event_kind(self, data: dict) -> str:
        """입력 데이터를 정규화된 이벤트 종류 문자열로 변환한다.

        반환값은 다음 중 하나여야 한다: 'prompt', 'response', 'unknown'.
        에이전트 고유의 hook 이름·스키마 해석은 이 메서드 안에서만 한다.
        """

    @classmethod
    @abstractmethod
    def install(cls) -> None:
        """이 어댑터에 해당하는 에이전트의 hook 시스템에 RAGent 호출을 등록한다.

        각 어댑터는 자신의 에이전트가 사용하는 hook 설정 파일 위치, 스키마,
        이벤트 이름을 알고 처리한다. 등록되는 hook 명령에는 어댑터 선택자가
        명시적으로 박혀 있어야 한다 — RAGent 시작 시점에 어떤 어댑터를 쓸지
        결정하는 신호이기 때문이다. 명령 문자열은 build_hook_command()로
        만들고(--adapter 인자로 선택자 전달, 셸 비종속), 멱등 재설치를 위한
        기존 hook 식별은 is_ragent_hook()을 쓴다.

        멱등해야 한다 — 두 번 호출해도 hook이 중복 등록되지 않아야 한다.
        """

    @staticmethod
    def build_hook_command(adapter_name: str) -> str:
        """호스트에 등록할, 셸 비종속 hook 명령 문자열을 만든다.

        셸 종속 문법(VAR=value 프리픽스 등)을 쓰지 않아 cmd(Windows)/sh(POSIX)
        양쪽에서 동작한다. 어댑터 선택자는 --adapter 인자로 명령에 직접 박는다.
        frozen(PyInstaller) 빌드는 실행 파일과 같은 디렉터리의 ragent-hook exe 를,
        소스 모드는 활성 파이썬의 `-m ragent` 를 가리킨다(ragent 패키지가 import
        가능해야 함 = `pip install -e .`). 공백이 든 경로를 위해 실행 파일은
        항상 따옴표로 감싼다.
        """
        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            exe_name = f"{HOOK_EXE}.exe" if sys.platform == "win32" else HOOK_EXE
            return f'"{exe_dir / exe_name}" --adapter {adapter_name}'
        return f'"{sys.executable}" -m ragent --adapter {adapter_name}'

    @staticmethod
    def is_ragent_hook(command: str) -> bool:
        """주어진 명령이 (버전 불문) RAGent hook 인지 판별한다(멱등 재설치용).

        소스 모드 명령(`-m ragent`)과 frozen 모드 명령(`ragent-hook` exe)을 모두
        인식한다. 구버전 명령(PYTHONPATH/RAGENT_ADAPTER 프리픽스가 붙은
        `-m ragent`) 역시 `-m ragent` 토큰으로 함께 제거된다.
        """
        return "-m ragent" in command or HOOK_EXE in command

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
        elif kind == "unknown":
            logger = logging.getLogger("ragent")
            logger.warning(
                "Adapter %s could not resolve event from data",
                type(self).__name__,
            )
        else:
            raise ValueError(
                f"_resolve_event_kind returned invalid value: {kind!r}. "
                f"Must be one of: prompt, response, unknown."
            )
