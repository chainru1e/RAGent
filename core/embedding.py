# ════════════════════════════════════════════════════════════════════════════
# core/embedding.py - 임베딩
# ════════════════════════════════════════════════════════════════════════════

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import Dict, List
from collections import Counter
import re


class HybridEmbedding:
    def __init__(self, dense_model_name="all-MiniLM-L6-v2"):
        print("✅ HybridEmbedding 초기화")

        self.dense_model = SentenceTransformer(dense_model_name)

        self.bm25 = None
        self.bm25_trained = False
        self.tokenized_docs = []
        self.vocabulary = {}

    def get_dense_embedding(self, text: str) -> np.ndarray:
        return self.dense_model.encode(text, convert_to_numpy=True).astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.dense_model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9가-힣_]', ' ', text)
        return text.split()

    def train_bm25(self, documents: List[str]):
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]

        vocab = set()
        for tokens in self.tokenized_docs:
            vocab.update(tokens)

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab))}
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.bm25_trained = True

        print(f"✅ BM25 학습 완료 ({len(self.vocabulary)} vocab)")

    # 🔥 핵심: Sparse Vector 생성

    def get_sparse_vector(self, text: str) -> Dict:
        tokens = self._tokenize(text)

        token_counts = Counter(tokens)

        indices = []
        values = []

        for token, count in token_counts.items():
            if token in self.vocabulary:
                indices.append(self.vocabulary[token])
                values.append(float(count))  # 빈도 기반

        return {
            "indices": indices,
            "values": values
        }