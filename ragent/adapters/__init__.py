"""어댑터 레지스트리와 팩토리.

환경변수 RAGENT_ADAPTER로 어댑터를 선택할 수 있으며, 미지정 시 'claude_code'를 사용한다.
"""

import os

from ragent.adapters.base import BaseAdapter
from ragent.adapters.claude_code import ClaudeCodeAdapter
from ragent.adapters.codex import CodexAdapter

ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    "claude_code": ClaudeCodeAdapter,
    "codex": CodexAdapter,
}


def get_adapter(name: str | None) -> BaseAdapter:
    """이름으로 어댑터 인스턴스를 얻는다.

    name이 None이거나 빈 문자열이면 환경변수 RAGENT_ADAPTER를 사용하고,
    그것도 없으면 기본값 'claude_code'를 사용한다.
    등록되지 않은 이름이면 ValueError를 발생시킨다.
    """
    if not name:
        name = os.environ.get("RAGENT_ADAPTER") or "claude_code"

    cls = ADAPTER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(ADAPTER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown adapter: {name!r}. Available adapters: {available}"
        )
    return cls()


__all__ = [
    "ADAPTER_REGISTRY",
    "BaseAdapter",
    "ClaudeCodeAdapter",
    "CodexAdapter",
    "get_adapter",
]
