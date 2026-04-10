# ════════════════════════════════════════════════════════════════════════════
# core/qdrant_storage.py - Qdrant 벡터 저장소
# ════════════════════════════════════════════════════════════════════════════

import numpy as np
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    SparseVectorParams,
    Distance,
    NamedVector,
    SparseVector
)
from core.embedding import HybridEmbedding


class QdrantStorage:
    def __init__(self, collection_name="rag_collection", path="./qdrant_storage"):
        self.collection_name = collection_name
        self.client = QdrantClient(path=path)

        self._init_collection()

    def _init_collection(self):
        try:
            self.client.get_collection(self.collection_name)
            print("✅ 기존 컬렉션 사용")
        except:
            print("🚀 새 컬렉션 생성")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=384, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )

    # 🔥 Dense + Sparse 같이 저장
    def add_point(self, chunk_id: str, dense_vector, sparse_vector, text: str):
        point = PointStruct(
            id=self._string_to_id(chunk_id),
            vector={
                "dense": dense_vector.tolist(),
                "sparse": sparse_vector
            },
            payload={"text": text, "chunk_id": chunk_id}
        )

        self.client.upsert(self.collection_name, [point])

    def add_points_batch(self, chunks: List[Dict]):
        points = []

        for chunk in chunks:
            points.append(
                PointStruct(
                    id=self._string_to_id(chunk["chunk_id"]),
                    vector={
                        "dense": chunk["dense"].tolist(),
                        "sparse": chunk["sparse"]
                    },
                    payload={"text": chunk["text"]}
                )
            )

        self.client.upsert(self.collection_name, points)
        return len(points)
    
    def get_stats(self) -> Dict:
        try:
            info = self.client.get_collection(self.collection_name)

            return {
                "collection": self.collection_name,
                "total_points": info.points_count,
                "status": str(info.status)
            }
        except Exception as e:
            return {
                "collection": self.collection_name,
                "error": str(e)
            }
        
    def _string_to_id(self, s: str) -> int:
        return int(hash(s) & 0x7FFFFFFF)

    # 🔥 진짜 Hybrid 검색
    def search(self, dense_vector, top_k=20):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", dense_vector.tolist()),
            limit=top_k
        )

        return [
            (r.payload["text"], r.score)
            for r in results
        ]