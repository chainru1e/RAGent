"""정규화된 메시지 표현.

각 어댑터(Claude Code, Codex, ...) 의 transcript 한 줄은 어댑터별 BaseParser
서브클래스에 의해 이 dataclass 로 번역된다. 다운스트림 컴포넌트(Chunker 등)는
원래 어댑터의 raw 스키마에 직접 의존하지 않는다.

필드 결정 근거(Phase 5 §0-3 조사):
- role / content: chunking_modules._extract_turn_components 와 session_end 의
  turn-start 판정에서 읽힌다.
- content_type: 현재 어떤 다운스트림도 직접 읽지 않는다. 본문(content) 의
  '[text]' / '[tool_result]' / '[Write]' 같은 prefix tag 가 사실상 같은
  정보를 운반하기 때문. 미래의 어댑터(예: Codex) 가 prefix tag 컨벤션을
  공유하지 않을 수 있어 구조화된 필드 자리를 함께 둔다 — 기본 None.
- timestamp: 기존 parser dict 에 존재했으므로 정보 보존 차원에서 유지.
  현재 다운스트림에서 직접 소비되는 위치는 없다.
"""

from dataclasses import dataclass


@dataclass
class NormalizedMessage:
    role: str
    content: str
    content_type: str | None = None
    timestamp: str | None = None
