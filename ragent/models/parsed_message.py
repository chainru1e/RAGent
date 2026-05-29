"""정규화된 메시지 표현.

각 어댑터(Claude Code, Codex, ...) 의 transcript 한 줄은 어댑터별 BaseParser
서브클래스에 의해 이 dataclass 로 번역된다. 다운스트림 컴포넌트(Chunker 등)는
원래 어댑터의 raw 스키마에 직접 의존하지 않는다.

필드 결정 근거:
- role: chunking_modules._extract_turn_components / contextual_retrieval_modules
  .serialize_turn 에서 읽힌다.
- blocks: 한 메시지의 content 블록들을 구조 보존(타입 + 원시 input)한 채 들고
  다닌다. 청커가 이 리스트를 순회하며 text/thinking 은 문맥으로, tool_use 는
  코드/비코드로 분기한다. tool_use 의 input 은 raw dict 그대로 보존되어 키
  이름이 살아 있으므로, 줄 위치로 의미를 추측할 필요가 없다.
- content: blocks 가 도입되기 전의 평탄 문자열 표현. 아직 blocks 를 채우지
  않는 어댑터(Codex/Windsurf) 의 폴백 경로와, turn 해시(결정론적 UUID5) 및
  serialize_turn 의 입력으로 여전히 읽힌다. 전환기의 이중 표현이며 전 어댑터가
  blocks 로 이관되면 제거 대상이다.
- timestamp: 기존 parser dict 에 존재했으므로 정보 보존 차원에서 유지.
  현재 다운스트림에서 직접 소비되는 위치는 없다.
"""

from dataclasses import dataclass, field


@dataclass
class Block:
    """한 메시지 content 배열의 단일 블록을 구조 보존한 표현.

    - text / thinking 블록 → type + text.
    - tool_use 블록 → type="tool_use", name=툴명, input=원시 tool_input dict
      그대로(키 보존, 가공·삭제 금지). new_string 추출 / old_string 폐기 같은
      의미 판단은 파서가 아니라 청커가 수행한다.
    """

    type: str
    text: str | None = None
    name: str | None = None
    input: dict | None = None


@dataclass
class NormalizedMessage:
    role: str
    content: str
    blocks: list[Block] = field(default_factory=list)
    timestamp: str | None = None
