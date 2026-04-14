from ragent.config import QDRANT_DIR, DENSE_SIZE, ensure_dirs
from ragent.models.chunk import Chunk, ChunkMetaData
from ragent.models.vector import HybridVector
from ragent.models.intent import IntentCategory

from enum import Enum
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    SparseVectorParams,
    Distance,
    Prefetch,
    FusionQuery,
    Fusion
)

from ragent.models.chunk import Chunk

class QdrantStorage:
    def __init__(self, collection_name: str, path=QDRANT_DIR):
        ensure_dirs()
        self.collection_name = collection_name
        self.client = QdrantClient(path=path)
        self._init_collection()

    def _init_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=DENSE_SIZE, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )

    def add_point(self, chunk: Chunk):
        meta = chunk.metadata
        vector = chunk.vector  # HybridVector

        point = PointStruct(
            id=self._string_to_id(meta.chunk_id),
            vector={
                "dense": vector.dense.tolist(),
                "sparse": vector.sparse  # SparseVector 객체
            },
            payload={
                "text": chunk.payload,
                "chunk_id": meta.chunk_id,
                "parent_id": meta.parent_id,
                "file_path": meta.file_path,
                "type": meta.type.value if isinstance(meta.type, Enum) else meta.type
            }
        )

        self.client.upsert(self.collection_name, [point])

    def add_points_batch(self, chunks: list[Chunk]) -> int:
        points = []

        for chunk in chunks:
            meta = chunk.metadata
            vector = chunk.vector  # HybridVector

            points.append(
                PointStruct(
                    id=self._string_to_id(meta.chunk_id),
                    vector={
                        "dense": vector.dense.tolist(),
                        "sparse": vector.sparse
                    },
                    payload={
                        "text": chunk.payload,
                        "chunk_id": meta.chunk_id,
                        "parent_id": meta.parent_id,
                        "file_path": meta.file_path,
                        "type": meta.type.value if isinstance(meta.type, Enum) else meta.type
                    }
                )
            )

        self.client.upsert(self.collection_name, points)
        return len(points)

    def hybrid_search(self, query_vector: HybridVector, limit: int = 5) -> list[Chunk]:
        """
        HybridVector를 입력받아 RRF 하이브리드 검색을 수행하고, 
        결과를 Chunk 객체 리스트로 반환합니다.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # 1. Dense Vector 검색 (numpy 배열을 list로 변환하여 전달)
                Prefetch(
                    query=query_vector.dense.tolist(),
                    using="dense",
                    limit=limit
                ),
                # 2. Sparse Vector 검색 (SparseVector 객체 그대로 전달)
                Prefetch(
                    query=query_vector.sparse, 
                    using="sparse",
                    limit=limit
                )
            ],
            # 3. RRF 알고리즘을 통한 점수 융합
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True
        )

        search_results = []
        for point in results.points:
            payload = point.payload
            
            # 저장할 때 meta.type.value 로 저장했으므로, 다시 Enum 객체로 복원합니다.
            intent_type = None
            if payload.get("type"):
                try:
                    intent_type = IntentCategory(payload["type"])
                except ValueError:
                    # Enum에 없는 값일 경우를 대비한 안전 장치
                    intent_type = payload["type"] 

            # 메타데이터 객체 조립
            metadata = ChunkMetaData(
                chunk_id=payload.get("chunk_id"),
                parent_id=payload.get("parent_id"),
                file_path=payload.get("file_path"),
                type=intent_type
            )
            
            # 최종 Chunk 객체 조립 (검색 결과 반환용이므로 vector는 None으로 처리)
            chunk = Chunk(
                metadata=metadata,
                payload=payload.get("text"),
                vector=None 
            )
            
            search_results.append(chunk)
            
        return search_results
    
    def get_stats(self) -> dict:
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
    
    def close(self):
        self.client.close()