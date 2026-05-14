# ================================================================
# LLM Intent Classifier (Google Gemini Edition)
# ================================================================
# 설치: pip install google-generativeai pydantic python-dotenv
# ================================================================

import json

from ragent.models.intent import IntentCategory, ClassificationResult
from ragent.llm_client import LLMClient


# ================================================================
# LLM 분류기
# ================================================================

class IntentClassifier:
    """LLMClient(OpenAI 호환 로컬 API)를 사용한 인텐트 분류기"""

    def __init__(self):
        system_prompt = """
            당신은 코딩 질문 분류기입니다.
            사용자의 질문을 분석하여 다음 4가지 카테고리 중 정확히 하나로 분류하세요:

            1. CODE_GENERATION - 새로운 코드/함수/클래스/API 작성 요청
            2. CODE_REFACTORING - 기존 코드 개선, 최적화, 리팩토링 요청
            3. CODE_DEBUGGING - 에러, 버그, 오류 해결 요청
            4. SIMPLE_QUESTION - 프로그래밍 개념, 설명, 비교 질문

            반드시 아래 JSON 형식으로만 응답하세요. 다른 설명은 추가하지 마세요:
            {"category": "카테고리명", "confidence": 0.0~1.0, "reasoning": "판단 이유"}
        """

        self.llm_client = LLMClient(system_prompt=system_prompt)

    def classify(self, query: str) -> ClassificationResult:
        """LLMClient를 호출하여 인텐트 분류"""

        try:
            response = self.llm_client.ask(query, temperature=0.1)

            # LLMClient 내부 오류 처리 ([Error] 문자열로 시작하는 경우)
            if response.startswith("[Error]"):
                raise Exception(response)
            
            result = json.loads(response)

            return ClassificationResult(
                category=IntentCategory[result["category"]],
                confidence=float(result["confidence"]),
                method="llm_based",
                reasoning=result.get("reasoning", "로컬 LLM 판단")
            )

        except Exception as e:
            return ClassificationResult(
                category=IntentCategory.SIMPLE_QUESTION,
                confidence=0.0,
                method="llm_based",
                reasoning=f"LLM 오류: {str(e)}"
            )
