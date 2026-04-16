"""
retriever.py
────────────
Two-Phase Intent-Aware Retriever.
Phase 1: 의도 매칭 필터 검색
Phase 2: 글로벌 전체 검색
Merge:   의도 매칭 문서에 score boost 적용
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from vector_store import VectorStore


class IntentAwareRetriever:
    """
    NCA 투영 모델과 VectorStore를 결합한 검색기.

    사용 흐름:
        1. query embedding (768D) → NCA 모델 → projected (128D)
        2. Phase 1: intent 필터 검색
        3. Phase 2: 글로벌 검색
        4. 결과 병합 + intent boost
    """

    def __init__(self, model: nn.Module,
                 vector_store: VectorStore,
                 device: torch.device,
                 intent_boost: float = 1.2,
                 top_k: int = 5):
        self.model = model
        self.model.eval()
        self.store = vector_store
        self.device = device
        self.intent_boost = intent_boost
        self.top_k = top_k

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """768D 임베딩을 128D로 투영합니다."""
        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x = x.to(self.device)
            projected = self.model(x)
            return projected.cpu().numpy().squeeze()

    def retrieve(self, query_embedding: np.ndarray,
                 predicted_intent: Optional[str] = None,
                 top_k: Optional[int] = None) -> list[dict]:
        """
        Two-Phase 검색을 수행합니다.

        Args:
            query_embedding: (768,) 원본 쿼리 임베딩
            predicted_intent: 예측된 사용자 의도 (None이면 글로벌 검색만)
            top_k: 반환할 문서 수 (None이면 self.top_k 사용)

        Returns:
            [{"doc_id", "text", "intent", "score"}, ...]
        """
        k = top_k or self.top_k
        query_proj = self.project(query_embedding)

        # ── Phase 1: Intent 필터 검색 ──
        phase1_results = []
        if predicted_intent:
            phase1_results = self.store.search(
                query_proj, top_k=k, intent_filter=predicted_intent
            )

        # ── Phase 2: 글로벌 검색 ──
        phase2_results = self.store.search(
            query_proj, top_k=k, intent_filter=None
        )

        # ── Merge: 중복 제거 + Intent Boost ──
        merged = {}
        for doc in phase1_results + phase2_results:
            doc_id = doc["doc_id"]
            if doc_id not in merged:
                merged[doc_id] = doc.copy()
            else:
                # 더 높은 점수 유지
                merged[doc_id]["score"] = max(
                    merged[doc_id]["score"], doc["score"]
                )

        # 의도 매칭 문서에 부스트 적용
        if predicted_intent:
            for doc in merged.values():
                if doc["intent"] == predicted_intent:
                    doc["score"] *= self.intent_boost

        # 점수 내림차순 정렬 후 상위 K개 반환
        final = sorted(merged.values(),
                       key=lambda x: x["score"],
                       reverse=True)
        return final[:k]
