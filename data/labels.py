# embedding_adapter/data/labels.py

from enum import Enum

class IntentCategory(Enum):
    CODE_GENERATION  = "CODE_GENERATION"
    CODE_REFACTORING = "CODE_REFACTORING"
    CODE_DEBUGGING   = "CODE_DEBUGGING"
    SIMPLE_QUESTION  = "SIMPLE_QUESTION"
    NO_RAG           = "NO_RAG"

LABEL2IDX = {
    c.value: i
    for i, c in enumerate(sorted(IntentCategory, key=lambda x: x.value))
}

IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}