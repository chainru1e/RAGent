"""Turn 단위 Contextual Retrieval 전처리.

각 코드 청크 앞에 LLM 이 생성한 짧은 맥락 문장(prefix) 을 붙여 임베딩 검색
품질을 높이는 전처리 모듈. "전체 문서" 는 현재 turn 전체이고, "대상" 은 한
개의 코드 청크이며, 청크가 그 turn 안에서 어떤 의도/맥락으로 등장했는지
한 문장으로 설명하게 한다.

KV cache 친화성: system prompt 는 호출 간 고정. user prompt 는 turn 텍스트를
앞쪽에 두고(같은 turn 안의 청크들 사이에 byte 동일), 청크 텍스트와 지시문을
뒤쪽에 둔다. 따라서 system + turn 텍스트까지가 동일 prefix 로 캐시 가능.

실패 격리: generate_prefix 는 예외/빈 응답/타임아웃 시 None 을 반환하며 절대
예외를 밖으로 던지지 않는다. index_turn 은 None 인 청크를 prefix 없이 진행.
"""

import logging
import re

from ragent.llm_client import LLMClient
from ragent.models.parsed_message import NormalizedMessage

logger = logging.getLogger("ragent")


# turn 직렬화 길이 상한. 초과 시 꼬리쪽을 잘라낸다. n_ctx 가 작은 환경에서는
# LLM 측에서 컨텍스트 초과로 실패할 수 있으나, 그 경우 generate_prefix 가
# None 을 반환하므로 index_turn 은 prefix 없이 계속 진행한다.
MAX_TURN_CHARS = 12000

# prefix 한 개의 최대 길이(문자수). 약 100 토큰 상당.
MAX_PREFIX_CHARS = 400


SYSTEM_PROMPT = """
    You are a coding context summarizer.
    Given a conversation turn (the document) and one extracted code chunk
    from that turn, write one short sentence that explains how the chunk
    fits into the turn — its intent, or what sub-task it serves.

    Do not restate or summarize the chunk's contents.
    Add only context that is missing from the chunk itself (user intent,
    which step of the task this chunk belongs to).
    Reply with the sentence only, no preamble.
"""


# turn_text 가 user prompt 맨 앞에 오도록 배치. 청크와 지시문은 뒤로.
USER_PROMPT_TEMPLATE = """<document>
{turn_text}
</document>

Here is the chunk we want to situate within the document:
<chunk>
{chunk_text}
</chunk>

Give one short sentence describing how this chunk fits within the turn
(its purpose or the user intent it serves), not what its contents are.
Reply with the sentence only."""


# LLM 출력의 첫 줄에 흔히 붙는 preamble 패턴
_PREAMBLE_PATTERNS = [
    re.compile(r"^\s*here(?:'s| is)[^\n]*:\s*", re.IGNORECASE),
    re.compile(r"^\s*sentence\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*context\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*summary\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE),
]


def serialize_turn(turn: list[NormalizedMessage]) -> str:
    """turn 메시지 리스트를 단일 문자열로 평탄화한다.

    role 표시(USER/ASSISTANT) 와 함께 각 메시지 본문을 누적한다. content 에는
    이미 [text]/[Write]/[tool_result] 등 prefix tag 가 박혀 있으므로 그대로
    유지한다. 결과가 MAX_TURN_CHARS 를 넘으면 꼬리쪽을 잘라낸다.
    """
    parts = []
    for msg in turn:
        role = (msg.role or "unknown").upper()
        content = (msg.content or "").strip()
        if not content:
            continue
        parts.append(f"[{role}]\n{content}")

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

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_prefix(self, source_turn_text: str, chunk_text: str) -> str | None:
        """단일 청크에 대한 맥락 문장을 생성한다.

        실패(예외/빈 응답/타임아웃) 시 None 을 반환하고 예외는 밖으로 던지지
        않는다. 호출자(index_turn) 는 None 인 경우 prefix 없이 진행한다.
        """
        if not source_turn_text or not chunk_text:
            return None

        prompt = USER_PROMPT_TEMPLATE.format(
            turn_text=source_turn_text,
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
