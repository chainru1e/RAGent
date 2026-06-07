"""Groq API를 사용한 LLM-as-Judge 모듈.

검색된 청크가 쿼리와 얼마나 관련 있는지 1~5점으로 평가한다.
Groq 무료 티어 사용 (llama-3.1-8b-instant).
"""

import os
from groq import Groq

JUDGE_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are an expert relevance evaluator for a code assistant RAG system.
Your job is to judge whether a retrieved text chunk is relevant to a given search query.

Scoring criteria:
5 = Highly relevant: directly answers or addresses the query
4 = Mostly relevant: closely related, useful context
3 = Somewhat relevant: partially related
2 = Mostly irrelevant: loosely related at best
1 = Completely irrelevant: unrelated to the query

Respond with ONLY a single integer from 1 to 5. No explanation."""

USER_PROMPT_TEMPLATE = """Query: {query}

Retrieved chunk:
\"\"\"
{chunk}
\"\"\"

Relevance score (1-5):"""


class LLMJudge:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY 환경변수를 설정하거나 --api-key 인자를 전달하세요.")
        self.client = Groq(api_key=key)

    def score(self, query: str, chunk_text: str) -> int:
        """단일 청크에 대한 관련도 점수 반환 (1~5)."""
        if not chunk_text or not chunk_text.strip():
            return 1

        response = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=4,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        query=query,
                        chunk=chunk_text[:1500],
                    ),
                },
            ],
        )

        raw = response.choices[0].message.content.strip()
        try:
            score = int(raw[0])
            return max(1, min(5, score))
        except (ValueError, IndexError):
            return 1

    def score_batch(self, query: str, chunks: list[str]) -> list[int]:
        """청크 리스트에 대해 순서대로 점수 반환."""
        return [self.score(query, chunk) for chunk in chunks]
