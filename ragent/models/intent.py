from enum import Enum
from typing import Optional
from pydantic import BaseModel


class IntentCategory(Enum):
    """코딩 에이전트가 처리할 4가지 인텐트 카테고리"""
    CODE_GENERATION = "CODE_GENERATION"
    CODE_REFACTORING = "CODE_REFACTORING"
    CODE_DEBUGGING = "CODE_DEBUGGING"
    SIMPLE_QUESTION = "SIMPLE_QUESTION"


class ClassificationResult(BaseModel):
    """분류 결과를 담는 구조화된 모델"""
    category: IntentCategory
    confidence: float
    method: str                    # "llm_based"
    reasoning: Optional[str] = None

    class Config:
        use_enum_values = True
