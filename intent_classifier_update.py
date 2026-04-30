# ================================================================
# Hybrid Intent Classifier — Multi-label (Sigmoid) Edition
# ================================================================
# 설치: pip install google-generativeai pydantic python-dotenv
# ================================================================
# 핵심 변경:
#   Softmax(합=1, 하나만 선택) → Sigmoid(각 의도 독립, 합≠1)
#   ClassificationResult(단일) → MultiLabelResult(복수 활성)
#   "버그 고치고 최적화해줘" → DEBUG 0.92 + REFACTOR 0.87 동시 활성
# ================================================================

import os
import re
import json
import math
import sys
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
#import google.generativeai as genai
import ollama


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


ALL_INTENTS = list(IntentCategory)


class RulePattern:
    """규칙 기반 분류를 위한 패턴 정의"""

    def __init__(self, category: IntentCategory, keywords: list,
                 weight: float = 1.0, regex_patterns: list = None):
        self.category = category
        self.keywords = keywords
        self.weight = weight
        self.regex_patterns = regex_patterns if regex_patterns is not None else []


class MultiLabelResult(BaseModel):
    """
    Multi-label 분류 결과
    ─────────────────────────────────────────────────
    intent_scores  : 각 의도별 독립 확률 (Sigmoid, 합 ≠ 1)
    active_intents : threshold를 넘긴 의도 리스트 (0개~5개)
    method         : "rule_based" | "llm_based"
    reasoning      : 판단 근거
    """
    intent_scores: dict[str, float]
    active_intents: list[str]
    method: str
    reasoning: Optional[str] = None

    should_retrieve: bool = False  # 검색(RAG) 엔진 가동 여부
    should_index: bool = False     # 벡터 DB 저장 여부

    model_config = ConfigDict(use_enum_values=True)


# ================================================================
# Section 2: Sigmoid 유틸리티
# ================================================================

def sigmoid(x: float) -> float:
    """수치 안정 시그모이드 — overflow 방지"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


# ================================================================
# Section 3: 규칙 기반 분류기 — Multi-label (Sigmoid)
# ================================================================

class RuleBasedClassifier:
    """
    키워드·정규식 매칭 raw 점수를 Sigmoid에 독립 통과시켜
    각 의도마다 0.0~1.0 확률을 산출하는 분류기.
    Softmax와 달리 의도 간 점수가 서로를 깎지 않는다.
    """

    def __init__(self, activation_threshold: float = 0.5,
                 sigmoid_bias: float = 2.0):
        self.rules = self._build_rules()
        self.activation_threshold = activation_threshold
        self.sigmoid_bias = sigmoid_bias  # 최소 매칭량 기준선

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
                    r"(not\\s+working|doesn\\'?t\\s+work|won\\'?t\\s+run|failed\\s+to)"
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
                    r".*(이|가)\\s*뭐(야|예요|인가요)",
                    r"(what\\s+is|what\\s+are|what\\s+does)\\s+",
                    r"(explain|describe|tell\\s+me\\s+about)\\s+",
                    r"(difference\\s+between|compare)\\s+",
                    r"(how\\s+does|how\\s+to|how\\s+do)\\s+"
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
                    r"^(hi|hello|hey|howdy|yo)[\\s!.,]*$",
                    r"^(thanks|thank\\s+you|thx|cheers)[\\s!.,]*$",
                    r"^(bye|goodbye|good\\s+night|see\\s+you)[\\s!.,]*$"
                ]
            ),
        ]

    def classify(self, query: str) -> MultiLabelResult:
        """
        각 의도에 대해 독립적으로 raw 점수 산출 후
        Sigmoid(score - bias)를 적용하여 독립 확률 변환.

        ┌─────────────────────────────────────────────────┐
        │  Softmax:  P(i) = exp(s_i) / Σ exp(s_k)        │
        │           → 합 = 1, 의도끼리 점수를 깎음         │
        │                                                 │
        │  Sigmoid:  P(i) = 1 / (1 + exp(-(s_i - bias))) │
        │           → 각 의도 독립, 동시 활성 가능          │
        └─────────────────────────────────────────────────┘
        """
        raw_scores: dict[str, float] = {}

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

            raw_scores[rule.category.value] = score

        # ── 각 의도에 독립 Sigmoid 적용 ──
        intent_scores = {}
        for intent_name in [c.value for c in ALL_INTENTS]:
            raw = raw_scores.get(intent_name, 0.0)
            intent_scores[intent_name] = round(
                sigmoid(raw - self.sigmoid_bias), 4
            )

        # ── threshold 이상인 의도 모두 활성화 ──
        active = [
            name for name, prob in intent_scores.items()
            if prob >= self.activation_threshold
        ]

        return MultiLabelResult(
            intent_scores=intent_scores,
            active_intents=active,
            method="rule_based",
            reasoning=(
                "raw_scores={"
                + ", ".join(
                    f"{k}: {v:.1f}" for k, v in raw_scores.items() if v > 0
                )
                + "}"
            )
        )


# ================================================================
# Section 4: LLM 분류기 — Multi-label (Sigmoid) Prompt
# ================================================================

# class LLMClassifier:
#     """
#     Google Gemini를 이용한 Multi-label 인텐트 분류기.
#     프롬프트에서 각 의도를 독립 판정하도록 지시하여
#     Softmax 제약(합=1) 없이 복합 의도를 포착한다.
#     """

#     def __init__(self, api_key: str):
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel(
#             model_name="gemini-2.5-flash",
#             generation_config={
#                 "response_mime_type": "application/json",
#                 "temperature": 0.1
#             }
#         )

#     def classify(self, query: str) -> MultiLabelResult:
#         """Gemini API를 호출하여 Multi-label 인텐트 분류"""

#         prompt = f"""당신은 코딩 질문의 **다중 의도 분류기**입니다.

# 사용자의 질문에는 여러 의도가 **동시에** 존재할 수 있습니다.
# 아래 5가지 의도 각각에 대해 **독립적으로** 해당 확률(0.0~1.0)을 판정하세요.

# - CODE_GENERATION  : 새로운 코드/함수/클래스/API 작성 요청
# - CODE_REFACTORING : 기존 코드 개선, 최적화, 리팩토링 요청
# - CODE_DEBUGGING   : 에러, 버그, 오류 해결 요청
# - SIMPLE_QUESTION  : 프로그래밍 개념, 설명, 비교 질문
# - NO_RAG           : 인사, 잡담, 감사 등 코딩과 무관한 대화

# **중요 — Sigmoid 독립 판정 원칙**:
# 각 점수는 서로 독립입니다. 합이 1이 될 필요가 없습니다.
# 예) "이 코드 버그 고치고 더 빠르게 최적화해줘"
#     → CODE_DEBUGGING=0.9, CODE_REFACTORING=0.85 동시에 높을 수 있음

# 반드시 아래 JSON 형식으로만 응답하세요:
# {{
#   "intent_scores": {{
#     "CODE_GENERATION": 0.0,
#     "CODE_REFACTORING": 0.0,
#     "CODE_DEBUGGING": 0.0,
#     "SIMPLE_QUESTION": 0.0,
#     "NO_RAG": 0.0
#   }},
#   "reasoning": "판단 이유"
# }}

# 사용자 질문: {query}"""

#         try:
#             response = self.model.generate_content(prompt)
#             result = json.loads(response.text)

#             scores = result["intent_scores"]
#             scores = {k: round(float(v), 4) for k, v in scores.items()}

#             active = [
#                 k for k, v in scores.items()
#                 if v >= 0.5
#             ]

#             return MultiLabelResult(
#                 intent_scores=scores,
#                 active_intents=active,
#                 method="llm_based",
#                 reasoning=result.get("reasoning", "Gemini 판단")
#             )

#         except Exception as e:
#             fallback = {c.value: 0.0 for c in ALL_INTENTS}
#             return MultiLabelResult(
#                 intent_scores=fallback,
#                 active_intents=[],
#                 method="llm_based",
#                 reasoning=f"Gemini API 오류: {str(e)}"
#             )
class LLMClassifier:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name  # 로컬에 설치된 모델명 (예: llama3, gemma2)

    def classify(self, query: str) -> MultiLabelResult:
        prompt = f"""사용자 질문의 의도를 분석하여 JSON으로 답하세요. 
        의도: CODE_GENERATION, CODE_REFACTORING, CODE_DEBUGGING, SIMPLE_QUESTION, NO_RAG
        형식: {{"intent_scores": {{"의도명": 0.0}}, "reasoning": "이유"}}
        질문: {query}"""

        try:
            # Ollama 로컬 호출 (format='json'으로 결과 강제)
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            
            result = json.loads(response['message']['content'])
            scores = {k: round(float(v), 4) for k, v in result["intent_scores"].items()}
            active = [k for k, v in scores.items() if v >= 0.5]

            return MultiLabelResult(
                intent_scores=scores,
                active_intents=active,
                method=f"local_llm({self.model_name})",
                reasoning=result.get("reasoning", "로컬 LLM 판단")
            )
        except Exception as e:
            return MultiLabelResult(intent_scores={}, active_intents=[], method="error", reasoning=str(e))



# ================================================================
# Section 4.5: RAG 동작 정책 (Policy)
# ================================================================

class RAGPolicy:
    """의도별 RAG 실행 정책 정의"""
    POLICY_MAP = {
        "CODE_GENERATION":  (False, True),  # 생성은 검색 X, 결과는 저장 O
        "CODE_REFACTORING": (True, True),   # 개선은 기존 코드 참고 필수
        "CODE_DEBUGGING":   (True, True),   # 에러는 과거 기록 참고 필수
        "SIMPLE_QUESTION":  (True, False),  # 질문은 검색은 하되, 굳이 저장 X
        "NO_RAG":           (False, False)  # 잡담은 둘 다 안 함
    }

    @staticmethod
    def get_action(active_intents: list[str]):
        # 여러 의도가 섞여있을 경우, 하나라도 True면 True로 판단 (OR 연산)
        should_retrieve = any(RAGPolicy.POLICY_MAP.get(i, (False, False))[0] for i in active_intents)
        should_index = any(RAGPolicy.POLICY_MAP.get(i, (False, False))[1] for i in active_intents)
        return should_retrieve, should_index
    

# ================================================================
# Section 5: 하이브리드 분류기 — Multi-label
# ================================================================

# class HybridClassifier:
#     """
#     Rule-Based(Sigmoid)를 먼저 시도하고,
#     활성 의도가 없거나 최대 확률이 낮으면 Gemini에게 위임하는
#     하이브리드 Multi-label 분류기.
#     """

#     def __init__(self, api_key: str,
#                  rule_confidence_threshold: float = 0.6,
#                  activation_threshold: float = 0.5):
#         self.rule_classifier = RuleBasedClassifier(
#             activation_threshold=activation_threshold
#         )
#         self.llm_classifier = LLMClassifier(api_key=api_key)
#         self.rule_confidence_threshold = rule_confidence_threshold

class HybridClassifier:
    def __init__(self, model_name: str = "llama3", rule_confidence_threshold: float = 0.6): 
        # 1. 인자에 rule_confidence_threshold를 추가하고
        self.rule_classifier = RuleBasedClassifier()
        self.llm_classifier = LLMClassifier(model_name=model_name)
        # 2. 아래와 같이 self에 저장해줘야 classify에서 쓸 수 있습니다.
        self.rule_confidence_threshold = rule_confidence_threshold 

    def classify(self, query: str) -> MultiLabelResult:
        # ── 1단계: Rule-Based 먼저 ──
        rule_result = self.rule_classifier.classify(query)

        max_score = (
            max(rule_result.intent_scores.values())
            if rule_result.intent_scores else 0.0
        )

        # 이제 self.rule_confidence_threshold를 정상적으로 참조합니다.
        if rule_result.active_intents and max_score >= self.rule_confidence_threshold:
            final_result = rule_result
        else:
            # ── 2단계: 로컬 LLM 호출 ──
            final_result = self.llm_classifier.classify(query)

        # ── 3단계: 정책(Policy) 적용 ──
        retrieve, index = RAGPolicy.get_action(final_result.active_intents)
        final_result.should_retrieve = retrieve
        final_result.should_index = index

        return final_result



# ================================================================
# Section 6: 메인 실행 및 올인원 래퍼 함수
# ================================================================

def smart_ask(query: str, classifier: HybridClassifier):
    """
    승준님이 호출만 하면 되는 '올인원' 대리인 함수입니다.
    이 안에서 분류, 검색 판단, 저장 판단이 모두 일어납니다.
    """
    # 1. 알아서 의도와 행동 지침을 가져옴
    result = classifier.classify(query)
    
    print(f"\n[분석 완료] 의도: {result.active_intents if result.active_intents else '없음'} ({result.method})")
    
    # 2. 결과에 적힌 대로 행동 (나중에 실제 RAG 코드를 이 안에 넣으면 됩니다)
    if result.should_retrieve:
        print("[동작] RAG 엔진 가동: 과거 대화 및 지식을 검색합니다.")
    else:
        print("[동작] RAG 검색 패스: 즉시 답변 생성을 시작합니다.")

    if result.should_index:
        print("[동작] 답변 생성 후 이 대화를 벡터 DB에 저장합니다.")
    else:
        print("[동작] 이 대화는 중요도가 낮아 저장하지 않습니다.")
        
    return result

# if __name__ == "__main__":
#     load_dotenv()
#     api_key = os.getenv("GEMINI_API_KEY")

#     if not api_key:
#         print("GEMINI_API_KEY가 설정되지 않았습니다.")
#         sys.exit(1)

#     classifier = HybridClassifier(
#         api_key=api_key,
#         rule_confidence_threshold=0.6,
#         activation_threshold=0.5
#     )
if __name__ == "__main__":
    # 로컬 LLM 환경이므로 API 키 설정 필요 없음
    classifier = HybridClassifier(model_name="llama3")

    print("=" * 60)
    print(" Smart RAG Agent - Intent & Action Controller (종료: quit)")
    print("=" * 60)

    while True:
        query = input("\n질문 입력: ").strip()

        if query.lower() in ["quit", "exit", "종료", "q"]:
            print("종료합니다.")
            break

        if not query:
            continue

        # --- [수정된 부분: 복잡한 로직 다 지우고 함수 하나만 호출!] ---
        smart_ask(query, classifier)