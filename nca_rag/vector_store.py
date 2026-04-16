"""
vector_store.py
───────────────
JSON 기반 벡터 저장소.
문서 수 ~10K까지는 이 구현으로 충분합니다.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional


class VectorStore:
    """
    JSON 파일 기반의 간이 벡터 저장소.

    구조:
        {
            "doc_id": {
                "text": "...",
                "intent": "...",
                "vector": [0.01, -0.23, ...]
            }
        }
    """

    def __init__(self, path: str = "vector_store.json"):
        self.path = Path(path)
        self.store: dict = {}
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                self.store = json.load(f)
            print(f"  📂 VectorStore loaded: {len(self.store)} docs ← {self.path}")

    def add(self, doc_id: str, text: str,
            intent: str, vector: list | np.ndarray) -> None:
        """문서 1건을 추가합니다."""
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        self.store[doc_id] = {
            "text": text,
            "intent": intent,
            "vector": vector,
        }

    def save(self) -> None:
        """현재 store를 JSON 파일로 저장합니다."""
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.store, f, ensure_ascii=False, indent=2)
        print(f"  💾 VectorStore saved: {len(self.store)} docs → {self.path}")

    def search(self, query_vector: np.ndarray,
               top_k: int = 5,
               intent_filter: Optional[str] = None) -> list[dict]:
        """
        코사인 유사도 기반 검색.

        Args:
            query_vector:  (D,) 쿼리 벡터
            top_k:         상위 K개 반환
            intent_filter: 특정 의도로 필터링 (None이면 전체 검색)

        Returns:
            [{"doc_id", "text", "intent", "score"}, ...]
        """
        if len(self.store) == 0:
            return []

        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)

        results = []
        for doc_id, doc in self.store.items():
            if intent_filter and doc["intent"] != intent_filter:
                continue

            doc_vec = np.array(doc["vector"], dtype=np.float32)
            doc_norm = doc_vec / (np.linalg.norm(doc_vec) + 1e-10)
            score = float(np.dot(query_norm, doc_norm))

            results.append({
                "doc_id": doc_id,
                "text": doc["text"],
                "intent": doc["intent"],
                "score": score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:
        return f"VectorStore(docs={len(self.store)}, path='{self.path}')"
