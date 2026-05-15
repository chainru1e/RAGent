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
            You are a coding question classifier.
            Analyze the user's question and classify it into exactly one of the following 4 categories:

            1. CODE_GENERATION - Requests to write new code/functions/classes/APIs
            2. CODE_REFACTORING - Requests to improve, optimize, or refactor existing code
            3. CODE_DEBUGGING - Requests to resolve errors, bugs, or issues
            4. SIMPLE_QUESTION - Questions about programming concepts, explanations, or comparisons

            You must respond ONLY in the following JSON format. Do not add any other explanations:
            {"category": "category_name", "confidence": 0.0~1.0, "reasoning": "reason for the decision"}
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
                reasoning=result.get("reasoning", "Local LLM decision")
            )

        except Exception as e:
            return ClassificationResult(
                category=IntentCategory.SIMPLE_QUESTION,
                confidence=0.0,
                method="llm_based",
                reasoning=f"LLM Error: {str(e)}"
            )
