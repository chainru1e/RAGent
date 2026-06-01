"""파일 단위 Contextual Retrieval 전처리.

각 코드 청크 앞에 LLM 이 생성한 짧은 맥락 문장(prefix) 을 붙여 임베딩 검색
품질을 높이는 전처리 모듈. "전체 문서(document)" 는 그 청크가 나온 **파일
전체**(큰 파일은 AST skeleton 으로 압축) 이고, "대상" 은 한 개의 코드 청크이며,
청크가 그 **파일** 안에서 어떤 역할/책임을 하는지(위치/맥락) 한 문장으로
설명하게 한다. 문서 자체는 index_turn 이 build_file_document 로 만들어 주입한다.

KV cache 친화성: system prompt 는 호출 간 고정. user prompt 는 파일 문서를
앞쪽에 두고(같은 파일에서 나온 청크들 사이에 byte 동일), 청크 텍스트와 지시문을
뒤쪽에 둔다. 따라서 system + 파일 문서까지가 동일 prefix 로 캐시 가능 — index_turn
이 같은 file_path 청크를 연속 호출하는 이유다.

실패 격리: generate_prefix 는 예외/빈 응답/타임아웃 시 None 을 반환하며 절대
예외를 밖으로 던지지 않는다. index_turn 은 None 인 청크를 prefix 없이 진행.
"""

import logging
import re

from ragent.llm_client import LLMClient
from ragent.models.parsed_message import Block, NormalizedMessage
from ragent.modules.chunking_modules import resolve_code_action

logger = logging.getLogger("ragent")


# turn 직렬화 길이 상한. 직렬화된 turn 프롬프트가 LLM 서버 n_ctx(4096 토큰)
# 안에 들어오도록 6000 으로 설정한다. 초과 시 꼬리쪽을 잘라낸다. 그래도 LLM
# 측에서 컨텍스트 초과로 실패하는 경우에는 generate_prefix 가 None 을 반환
# 하므로 index_turn 은 prefix 없이 계속 진행한다.
MAX_TURN_CHARS = 6000

# prefix 한 개의 최대 길이(문자수). 약 100 토큰 상당.
MAX_PREFIX_CHARS = 400


SYSTEM_PROMPT = """
    You are a coding context summarizer.
    Given a source file (the document) and one code chunk extracted from
    that file, write one short sentence that explains what role or
    responsibility the chunk has within the whole file — where it sits and
    what it is for.

    Do not restate or summarize the chunk's contents.
    Add only context that situates the chunk within the file (which part of
    the file/module it belongs to, what responsibility it serves).
    Reply with the sentence only, no preamble.
"""


# 파일 문서가 user prompt 맨 앞에 오도록 배치. 청크와 지시문은 뒤로 (KV-cache 친화).
USER_PROMPT_TEMPLATE = """<document>
{document_text}
</document>

Here is the chunk we want to situate within the file:
<chunk>
{chunk_text}
</chunk>

Give one short sentence describing what role this chunk plays within the
file (its position or responsibility), not what its contents are.
Reply with the sentence only."""


# LLM 출력의 첫 줄에 흔히 붙는 preamble 패턴
_PREAMBLE_PATTERNS = [
    re.compile(r"^\s*here(?:'s| is)[^\n]*:\s*", re.IGNORECASE),
    re.compile(r"^\s*sentence\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*context\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*summary\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE),
]


def _render_block(block: Block) -> str | None:
    """단일 Block 을 사람이 읽을 수 있는 prefix-tag 문자열로 렌더한다.

    - text/thinking → "[<type>]\\n<text>"
    - tool_use(코드 액션) → "[<tool_name>]\\n<file_path>\\n<code>".
      어느 input 키가 코드인지는 `resolve_code_action` 단일 정의에 위임한다.
    - tool_use(비코드/해석 불가: Read/Bash 등) → "[<tool_name>]\\n<input.values()>"
      로 generic dump 하여 컨텍스트를 보존한다.
    - 알 수 없는 타입 → None (스킵).
    """
    if block.type in ("text", "thinking"):
        return f"[{block.type}]\n{block.text or ''}"

    if block.type == "tool_use":
        name = block.name or "tool_use"
        action = resolve_code_action(block)
        if action is not None:
            file_path, code = action
            return f"[{name}]\n{file_path}\n{code}"
        # 비코드/해석 불가 tool_use: 원시 input 을 generic dump (컨텍스트 손실 방지)
        if isinstance(block.input, dict) and block.input:
            values = "\n".join(str(v) for v in block.input.values())
            return f"[{name}]\n{values}"
        return f"[{name}]"

    return None


def serialize_turn(turn: list[NormalizedMessage]) -> str:
    """turn 메시지 리스트를 단일 문자열로 평탄화한다.

    role 표시(USER/ASSISTANT) 와 함께 각 메시지 본문을 누적한다. blocks 가
    채워진 메시지는 블록을 순회해 렌더하고(코드 액션의 의미는 resolve_code_action
    단일 정의 경유), blocks 가 빈 메시지(미이관 어댑터/사용자 메시지) 는 기존
    content 경로로 폴백한다. content 에는 이미 [text]/[tool_result] 등 prefix tag
    가 박혀 있으므로 그대로 유지한다. 결과가 MAX_TURN_CHARS 를 넘으면 꼬리쪽을
    잘라낸다.
    """
    parts = []
    for msg in turn:
        role = (msg.role or "unknown").upper()
        if msg.blocks:
            rendered = [r for r in (_render_block(b) for b in msg.blocks) if r]
            body = "\n".join(rendered).strip()
        else:
            body = (msg.content or "").strip()
        if not body:
            continue
        parts.append(f"[{role}]\n{body}")

    joined = "\n\n".join(parts)

    if len(joined) > MAX_TURN_CHARS:
        joined = joined[:MAX_TURN_CHARS] + "\n[...truncated...]"
    return joined


def _clean_prefix(raw: str) -> str:
    """LLM 출력에서 preamble 을 떼고 길이 cap 을 적용한다."""
    text = raw.strip()
    if not text:
        return ""

    for pattern in _PREAMBLE_PATTERNS:
        text = pattern.sub("", text, count=1)

    text = text.strip().strip('"').strip("'").strip()

    if len(text) > MAX_PREFIX_CHARS:
        text = text[:MAX_PREFIX_CHARS].rstrip()
    return text


class ContextualEnricher:
    """코드 청크 앞에 붙일 맥락 문장을 LLM 으로 생성한다."""

    def __init__(self):
        self.llm_client = LLMClient(system_prompt=SYSTEM_PROMPT)

    def generate_prefix(self, document_text: str, chunk_text: str) -> str | None:
        """단일 청크에 대한 맥락 문장을 생성한다.

        document_text 는 그 청크가 나온 파일 문서(원문 또는 skeleton/truncated)
        이며, fallback 시에는 serialize_turn 결과가 들어올 수 있다.

        실패(예외/빈 응답/타임아웃) 시 None 을 반환하고 예외는 밖으로 던지지
        않는다. 호출자(index_turn) 는 None 인 경우 prefix 없이 진행한다.
        """
        if not document_text or not chunk_text:
            return None

        prompt = USER_PROMPT_TEMPLATE.format(
            document_text=document_text,
            chunk_text=chunk_text,
        )

        try:
            response = self.llm_client.ask(
                prompt,
                override_system_prompt=SYSTEM_PROMPT,
                temperature=0.2,
            )
        except Exception:
            logger.exception("contextual_enricher: LLM call raised")
            return None

        if not response or response.startswith("[Error]"):
            logger.warning("contextual_enricher: empty/error response: %r", response)
            return None

        cleaned = _clean_prefix(response)
        return cleaned or None
