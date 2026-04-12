from enum import Enum
from typing import Optional
from pydantic import BaseModel

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
