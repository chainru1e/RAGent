"""
=============================================================
Low-Rank NCA RAG Pipeline
=============================================================
임베딩 모델 의존성 없음.
외부에서 벡터를 직접 넣어주는 구조.

Dependencies:
    pip install numpy scikit-learn
=============================================================
"""

import json
import os
import pickle
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder


# =============================================================
# 1. NCA Transformer
# =============================================================

class NCATransformer:
    def __init__(self, n_components: int = 128, max_iter: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.nca = NeighborhoodComponentsAnalysis(
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
            verbose=1
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NCATransformer":
        print(f"[NCA] 학습 시작")
        print(f"  데이터: {len(X)}개")
        print(f"  {X.shape[1]}D -> {self.n_components}D")
        print(f"  Intent: {sorted(set(y))}")
        y_encoded = self.label_encoder.fit_transform(y)
        self.nca.fit(X, y_encoded)
        self.is_fitted = True
        print(f"[NCA] 학습 완료! 행렬 A: {self.nca.components_.shape}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        result = self.nca.transform(X)
        return result[0] if single else result

    def save(self, path: str = "nca_model.pkl"):
        if not self.is_fitted:
            raise RuntimeError("학습 안 된 모델은 저장 불가.")
        with open(path, "wb") as f:
            pickle.dump({
                "nca": self.nca,
                "label_encoder": self.label_encoder,
                "n_components": self.n_components,
                "is_fitted": True
            }, f)
        print(f"[NCA] 저장 -> {path}")

    def load(self, path: str = "nca_model.pkl") -> "NCATransformer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.nca = data["nca"]
        self.label_encoder = data["label_encoder"]
        self.n_components = data["n_components"]
        self.is_fitted = data["is_fitted"]
        print(f"[NCA] 로드 <- {path}")
        return self


# =============================================================
# 2. Vector Store (JSON 기반)
# =============================================================

class VectorStore:
    def __init__(self, dimension: int, metric: str = "cosine", store_path: str = "vector_store.json"):
        self.dimension = dimension
        self.metric = metric
        self.store_path = store_path
        self.vectors: Dict[str, Dict] = {}
        if os.path.exists(store_path):
            self._load()

    def add(self, vector: np.ndarray, metadata: Dict[str, Any], text: str = "") -> str:
        doc_id = str(uuid.uuid4())[:8]
        self.vectors[doc_id] = {
            "vector": vector.tolist(),
            "metadata": metadata,
            "text": text
        }
        return doc_id

    def add_batch(self, vectors: np.ndarray, metadatas: List[Dict], texts: List[str]) -> List[str]:
        return [self.add(v, m, t) for v, m, t in zip(vectors, metadatas, texts)]

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        boost_intent: Optional[str] = None,
        intent_boost: float = 1.0
    ) -> List[Dict]:
        if not self.vectors:
            return []
        query = np.array(query_vector)
        results = []
        for doc_id, doc in self.vectors.items():
            if filter_metadata:
                if any(doc["metadata"].get(k) != v for k, v in filter_metadata.items()):
                    continue
            stored_vec = np.array(doc["vector"])
            score = self._similarity(query, stored_vec)
            if boost_intent and doc["metadata"].get("intent") == boost_intent:
                score *= intent_boost
            results.append({
                "id": doc_id,
                "score": float(score),
                "text": doc["text"],
                "metadata": doc["metadata"]
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def two_phase_search(
        self,
        query_vector: np.ndarray,
        intent: str,
        top_k: int = 5,
        intent_boost: float = 1.2
    ) -> List[Dict]:
        phase1 = self.search(query_vector, top_k=top_k, filter_metadata={"intent": intent})
        phase2 = self.search(query_vector, top_k=top_k, boost_intent=intent, intent_boost=intent_boost)
        seen = set()
        merged = []
        for r in phase1 + phase2:
            if r["id"] not in seen:
                seen.add(r["id"])
                if r["metadata"].get("intent") == intent:
                    r["score"] *= intent_boost
                merged.append(r)
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    def save(self):
        data = {
            "config": {"dimension": self.dimension, "metric": self.metric},
            "vectors": self.vectors
        }
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Store] 저장 -> {self.store_path} ({len(self.vectors)}개)")

    def _load(self):
        with open(self.store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vectors = data.get("vectors", {})
        cfg = data.get("config", {})
        self.dimension = cfg.get("dimension", self.dimension)
        self.metric = cfg.get("metric", self.metric)
        print(f"[Store] 로드 <- {self.store_path} ({len(self.vectors)}개)")

    def clear(self):
        self.vectors = {}

    def count(self) -> int:
        return len(self.vectors)

    def get_stats(self) -> Dict:
        intents = {}
        for doc in self.vectors.values():
            i = doc["metadata"].get("intent", "UNKNOWN")
            intents[i] = intents.get(i, 0) + 1
        return {"total": self.count(), "dimension": self.dimension, "metric": self.metric, "intents": intents}

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "cosine":
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / norm) if norm else 0.0
        else:
            return float(1.0 / (1.0 + np.linalg.norm(a - b)))


# =============================================================
# 3. Pipeline
# =============================================================

class NCAPipeline:
    def __init__(
        self,
        n_components: int = 128,
        metric: str = "cosine",
        store_path: str = "vector_store.json",
        nca_path: str = "nca_model.pkl"
    ):
        self.nca = NCATransformer(n_components=n_components)
        self.nca_path = nca_path
        self.store = VectorStore(dimension=n_components, metric=metric, store_path=store_path)
        if os.path.exists(nca_path):
            self.nca.load(nca_path)

    def train(self, training_data_path: str) -> None:
        with open(training_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        X = np.array([item["vector"] for item in data])
        y = np.array([item["intent"] for item in data])
        self.nca.fit(X, y)
        self.nca.save(self.nca_path)

    def train_from_arrays(self, X: np.ndarray, y: np.ndarray) -> None:
        self.nca.fit(X, y)
        self.nca.save(self.nca_path)

    def store_one(self, vector: np.ndarray, metadata: Dict[str, Any], text: str = "") -> str:
        transformed = self.nca.transform(np.array(vector))
        return self.store.add(transformed, metadata, text)

    def store_batch(self, vectors: np.ndarray, metadatas: List[Dict], texts: List[str]) -> List[str]:
        transformed = self.nca.transform(np.array(vectors))
        return self.store.add_batch(transformed, metadatas, texts)

    def store_from_json(self, data_path: str) -> List[str]:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vectors = np.array([item["vector"] for item in data])
        metadatas = [item.get("metadata", {}) for item in data]
        texts = [item.get("text", "") for item in data]
        return self.store_batch(vectors, metadatas, texts)

    def search(
        self,
        query_vector: np.ndarray,
        intent: Optional[str] = None,
        top_k: int = 5,
        use_two_phase: bool = True,
        intent_boost: float = 1.2
    ) -> List[Dict]:
        q = self.nca.transform(np.array(query_vector))
        if intent and use_two_phase:
            return self.store.two_phase_search(q, intent, top_k, intent_boost)
        elif intent:
            return self.store.search(q, top_k, filter_metadata={"intent": intent})
        else:
            return self.store.search(q, top_k)

    def save(self):
        self.store.save()

    def get_stats(self) -> Dict:
        return {
            "nca_fitted": self.nca.is_fitted,
            "nca_dim": self.nca.n_components,
            "store": self.store.get_stats()
        }


# =============================================================
# 4. Demo
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  NCA RAG Pipeline - Demo")
    print("=" * 60)
    print()

    N_TRAIN = 200
    N_STORE = 20
    DIM = 768
    N_COMPONENTS = 128
    INTENTS = ["CODE_DEBUGGING", "CODE_GENERATION", "CODE_REFACTORING", "SIMPLE_QUESTION"]

    np.random.seed(42)

    # 1. 학습 데이터
    print("[1/5] 가짜 학습 데이터 생성...")
    X_train = np.random.randn(N_TRAIN, DIM).astype(np.float32)
    y_train = np.array([INTENTS[i % len(INTENTS)] for i in range(N_TRAIN)])
    print(f"  X: {X_train.shape}, y: {len(set(y_train))}개 intent")
    print()

    # 2. NCA 학습
    print("[2/5] NCA 학습...")
    pipe = NCAPipeline(n_components=N_COMPONENTS)
    pipe.train_from_arrays(X_train, y_train)
    print()

    # 3. 문서 저장
    print("[3/5] 문서 저장...")
    sample_texts = [
        "IndexError: list index out of range 에러 발생",
        "REST API 엔드포인트 생성해줘",
        "이 함수 리팩토링 해줘",
        "Python에서 리스트 정렬하는 법",
        "NullPointerException 해결 방법",
        "FastAPI CRUD 코드 만들어줘",
        "중복 코드 제거하고 싶어",
        "딕셔너리 사용법 알려줘",
        "TypeError: unsupported operand 에러",
        "WebSocket 서버 구현해줘",
        "클래스 상속 구조 개선",
        "for문 사용법",
        "segfault 디버깅 방법",
        "로그인 기능 구현 코드",
        "매직넘버 상수로 추출",
        "변수 스코프가 뭐야?",
        "메모리 릭 찾는 방법",
        "GraphQL 스키마 작성해줘",
        "긴 함수 분리하기",
        "람다 함수 설명해줘",
    ]
    for i in range(N_STORE):
        vec = np.random.randn(DIM).astype(np.float32)
        intent = INTENTS[i % len(INTENTS)]
        pipe.store_one(vec, metadata={"intent": intent, "lang": "python"}, text=sample_texts[i])
    pipe.save()
    print()

    # 4. 검색 테스트
    print("[4/5] 검색 테스트...")
    query = np.random.randn(DIM).astype(np.float32)

    print()
    print("  --- Two-Phase Search (intent=CODE_DEBUGGING) ---")
    results = pipe.search(query, intent="CODE_DEBUGGING", top_k=5, use_two_phase=True)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['text']}  (intent={r['metadata']['intent']})")

    print()
    print("  --- Global Search (no intent) ---")
    results_global = pipe.search(query, intent=None, top_k=5)
    for r in results_global:
        print(f"  [{r['score']:.4f}] {r['text']}  (intent={r['metadata']['intent']})")

    # 5. 통계
    print()
    print("[5/5] 파이프라인 상태:")
    stats = pipe.get_stats()
    print(f"  NCA fitted : {stats['nca_fitted']}")
    print(f"  NCA dim    : {stats['nca_dim']}D")
    print(f"  Store      : {stats['store']['total']}개 문서")
    print(f"  Intents    : {stats['store']['intents']}")
    print()
    print("Done!")
