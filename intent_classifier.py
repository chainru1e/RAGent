# ================================================================
# Hybrid Intent Classifier (Google Gemini Edition)
# ================================================================
# 설치: pip install google-generativeai pydantic python-dotenv
# ================================================================

import os
import re
import json
import sys
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai


# ================================================================
# Section 1: 데이터 모델 (Enum + Class + Pydantic)
# ================================================================

class IntentCategory(Enum):
    """코딩 에이전트가 처리할 5가지 인텐트 카테고리"""
    CODE_GENERATION = "CODE_GENERATION"
    CODE_REFACTORING = "CODE_REFACTORING"
    CODE_DEBUGGING = "CODE_DEBUGGING"
    SIMPLE_QUESTION = "SIMPLE_QUESTION"
    NO_RAG = "NO_RAG"


class RulePattern:
    """규칙 기반 분류를 위한 패턴 정의"""

    def __init__(self, category: IntentCategory, keywords: list,
                 weight: float = 1.0, regex_patterns: list = None):
        self.category = category
        self.keywords = keywords
        self.weight = weight
        self.regex_patterns = regex_patterns if regex_patterns is not None else []


class ClassificationResult(BaseModel):
    """분류 결과를 담는 구조화된 모델"""
    category: IntentCategory
    confidence: float
    method: str                    # "rule_based" 또는 "llm_based"
    reasoning: Optional[str] = None

    class Config:
        use_enum_values = True


# ================================================================
# Section 2: 규칙 기반 분류기 (방법 B) — 무료, 즉시 판단
# ================================================================

class RuleBasedClassifier:
    """키워드와 정규식으로 인텐트를 분류하는 로컬 분류기 (한/영 지원)"""

    def __init__(self):
        self.rules = self._build_rules()

    def _build_rules(self) -> list:
        return [
            # ── 코드 생성 ──
            RulePattern(
                category=IntentCategory.CODE_GENERATION,
                keywords=[
                    # 한국어
                    "만들어", "생성", "작성", "구현", "코딩", "개발", "짜줘", "작성해",
                    # English
                    "create", "generate", "write", "implement", "build", "develop",
                    "make", "code", "scaffold", "boilerplate"
                ],
                weight=2.0,
                regex_patterns=[
                    r"(만들어|생성|작성|구현).*(코드|함수|클래스|API|프로그램)",
                    r"(코드|함수|클래스|API|프로그램).*(만들어|생성|작성|구현)",
                    r"(create|generate|write|implement|build).*(code|function|class|api|program|script)",
                    r"(code|function|class|api|program|script).*(create|generate|write|implement|build)"
                ]
            ),
            # ── 코드 리팩토링 ──
            RulePattern(
                category=IntentCategory.CODE_REFACTORING,
                keywords=[
                    # 한국어
                    "리팩토링", "개선", "최적화", "클린코드", "수정", "변환", "바꿔", "깔끔",
                    # English
                    "refactor", "improve", "optimize", "clean", "restructure", "simplify",
                    "rewrite", "redesign", "convert", "migrate", "performance"
                ],
                weight=2.0,
                regex_patterns=[
                    r"(개선|최적화|리팩토링).*(코드|함수|성능)",
                    r"(코드|함수).*(개선|최적화|리팩토링|깔끔)",
                    r"(improve|optimize|refactor|clean\s*up).*(code|function|performance|logic)",
                    r"(code|function).*(improve|optimize|refactor|clean|simplify)"
                ]
            ),
            # ── 코드 디버깅 ──
            RulePattern(
                category=IntentCategory.CODE_DEBUGGING,
                keywords=[
                    # 한국어
                    "에러", "오류", "버그", "안됨", "안돼", "실패", "문제", "디버깅", "고쳐",
                    # English
                    "error", "bug", "debug", "fix", "broken", "crash", "fail", "issue",
                    "exception", "traceback", "stacktrace", "not working", "doesn't work"
                ],
                weight=2.5,
                regex_patterns=[
                    r"(에러|오류|버그).*(나|발생|해결|고쳐)",
                    r"(안\s*됨|안\s*돼|작동.*안|실행.*안)",
                    r"(TypeError|ValueError|SyntaxError|NameError|IndentationError)",
                    r"(KeyError|AttributeError|IndexError|ImportError|ModuleNotFoundError)",
                    r"(RuntimeError|ZeroDivisionError|FileNotFoundError|PermissionError)",
                    r"(error|bug|crash).*(fix|solve|resolve|help|debug)",
                    r"(fix|solve|resolve|debug).*(error|bug|crash|issue|problem)",
                    r"(not\s+working|doesn'?t\s+work|won'?t\s+run|failed\s+to)"
                ]
            ),
            # ── 단순 질문 ──
            RulePattern(
                category=IntentCategory.SIMPLE_QUESTION,
                keywords=[
                    # 한국어
                    "뭐야", "뭔가요", "알려줘", "설명", "차이", "비교", "장단점", "어떻게",
                    # English
                    "what is", "what are", "explain", "describe", "difference", "compare",
                    "how does", "how to", "why does", "pros and cons", "meaning", "definition"
                ],
                weight=1.5,
                regex_patterns=[
                    r"(뭐야|뭔가요|알려줘|설명해|차이.*뭐)",
                    r".*(이|가)\s*뭐(야|예요|인가요)",
                    r"(what\s+is|what\s+are|what\s+does)\s+",
                    r"(explain|describe|tell\s+me\s+about)\s+",
                    r"(difference\s+between|compare)\s+",
                    r"(how\s+does|how\s+to|how\s+do)\s+"
                ]
            ),
            # ── NO_RAG (인사/잡담) ──
            RulePattern(
                category=IntentCategory.NO_RAG,
                keywords=[
                    # 한국어
                    "안녕", "고마워", "감사", "ㅎㅎ", "ㅋㅋ", "반가워", "수고",
                    # English
                    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
                    "good morning", "good night", "cheers", "appreciate"
                ],
                weight=2.0,
                regex_patterns=[
                    r"^(안녕|하이|헬로)",
                    r"^(고마워|감사합니다|땡큐)",
                    r"^(hi|hello|hey|howdy|yo)[\s!.,]*$",
                    r"^(thanks|thank\s+you|thx|cheers)[\s!.,]*$",
                    r"^(bye|goodbye|good\s+night|see\s+you)[\s!.,]*$"
                ]
            ),
        ]

    def classify(self, query: str) -> ClassificationResult:
        """입력 쿼리를 규칙 기반으로 분류"""
        scores = {}

        for rule in self.rules:
            score = 0.0

            # 키워드 매칭 (대소문자 무시)
            for keyword in rule.keywords:
                if keyword in query.lower():
                    score += rule.weight

            # 정규식 매칭
            for pattern in rule.regex_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += rule.weight * 1.5

            if score > 0:
                scores[rule.category] = scores.get(rule.category, 0) + score

        # 매칭 없으면 기본값
        if not scores:
            return ClassificationResult(
                category=IntentCategory.SIMPLE_QUESTION,
                confidence=0.0,
                method="rule_based",
                reasoning="매칭되는 규칙 없음 -> 기본값 SIMPLE_QUESTION"
            )

        best_category = max(scores, key=scores.get)
        return ClassificationResult(
            category=best_category,
            confidence=scores[best_category],
            method="rule_based",
            reasoning=f"키워드/정규식 매칭 점수: {scores[best_category]:.1f}"
        )


# ================================================================
# Section 3: LLM 분류기 (방법 A) — Google Gemini API
# ================================================================

class LLMClassifier:
    """Google Gemini를 사용한 LLM 기반 인텐트 분류기"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )

    def classify(self, query: str) -> ClassificationResult:
        """Gemini API를 호출하여 인텐트 분류"""

        prompt = f"""당신은 코딩 질문 분류기입니다.
사용자의 질문을 분석하여 다음 5가지 카테고리 중 정확히 하나로 분류하세요:

1. CODE_GENERATION - 새로운 코드/함수/클래스/API 작성 요청
2. CODE_REFACTORING - 기존 코드 개선, 최적화, 리팩토링 요청
3. CODE_DEBUGGING - 에러, 버그, 오류 해결 요청
4. SIMPLE_QUESTION - 프로그래밍 개념, 설명, 비교 질문
5. NO_RAG - 인사, 잡담, 감사 등 코딩과 무관한 대화

반드시 아래 JSON 형식으로만 응답하세요:
{{"category": "카테고리명", "confidence": 0.0~1.0, "reasoning": "판단 이유"}}

사용자 질문: {query}"""

        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)

            return ClassificationResult(
                category=IntentCategory[result["category"]],
                confidence=float(result["confidence"]),
                method="llm_based",
                reasoning=result.get("reasoning", "Gemini 판단")
            )

        except Exception as e:
            return ClassificationResult(
                category=IntentCategory.SIMPLE_QUESTION,
                confidence=0.0,
                method="llm_based",
                reasoning=f"Gemini API 오류: {str(e)}"
            )


# ================================================================
# Section 4: 하이브리드 분류기 (방법 C) — Rule + LLM 합체 + 실행
# ================================================================

class HybridClassifier:
    """Rule-Based를 먼저 시도하고, 확신 없으면 Gemini에게 넘기는 하이브리드 분류기"""

    def __init__(self, api_key: str, confidence_threshold: float = 3.0):
        self.rule_classifier = RuleBasedClassifier()
        self.llm_classifier = LLMClassifier(api_key=api_key)
        self.confidence_threshold = confidence_threshold

    def classify(self, query: str) -> ClassificationResult:
        """
        하이브리드 분류 로직:
        1단계: Rule-Based로 빠르게 판단 시도
        2단계: 확신 부족하면 -> Gemini LLM에게 위임
        """

        # ── 1단계: Rule-Based 먼저 ──
        rule_result = self.rule_classifier.classify(query)

        if rule_result.confidence >= self.confidence_threshold:
            return rule_result

        # ── 2단계: Gemini LLM 호출 ──
        return self.llm_classifier.classify(query)


# ── 메인 실행 ──
if __name__ == "__main__":

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("GEMINI_API_KEY가 설정되지 않았습니다.")
        print(".env 파일에 GEMINI_API_KEY=your-key 를 추가하세요.")
        sys.exit(1)

    classifier = HybridClassifier(api_key=api_key, confidence_threshold=3.0)

    print("=" * 55)
    print("Hybrid Intent Classifier (종료: quit)")
    print("=" * 55)

    while True:
        query = input("\n질문 입력: ").strip()

        if query.lower() in ["quit", "exit", "종료", "q"]:
            print("종료합니다.")
            break

        if not query:
            continue

        result = classifier.classify(query)

        print(f"  분류: {result.category}")
        print(f"  확신: {result.confidence}")
        print(f"  방법: {result.method}")
        print(f"  이유: {result.reasoning}")
