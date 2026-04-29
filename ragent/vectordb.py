from ragent.config import QDRANT_DIR, SHORT_DENSE_SIZE, LONG_DENSE_SIZE, ensure_dirs
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
                    "dense_short": VectorParams(size=SHORT_DENSE_SIZE, distance=Distance.COSINE),
                    "dense_long": VectorParams(size=LONG_DENSE_SIZE, distance=Distance.COSINE)
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
                "dense_short": vector.dense[:SHORT_DENSE_SIZE].tolist(),
                "dense_long": vector.dense.tolist(),
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
                        "dense_short": vector.dense[:SHORT_DENSE_SIZE].tolist(),
                        "dense_long": vector.dense.tolist(),
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
    
    def payload_to_chunk(self, payload: dict) -> Chunk:
        """Qdrant Payload를 Chunk 객체로 변환합니다."""
        intent_type = None
        if payload.get("type"):
            try:
                intent_type = IntentCategory(payload.get("type"))
            except ValueError:
                intent_type = payload.get("type") 

        metadata = ChunkMetaData(
            chunk_id=payload.get("chunk_id"),
            parent_id=payload.get("parent_id"),
            file_path=payload.get("file_path"),
            type=intent_type
        )
        
        return Chunk(
            metadata=metadata,
            payload=payload.get("text"),
            vector=None 
        )

    def staged_hybrid_search(self, query_vector: HybridVector, limit: int = 5) -> list[Chunk]:
        """
        2단계 dense 검색과 sparse 검색을 RRF로 융합하는 하이브리드 검색을 수행한다.

        dense 검색은 short vector로 후보군을 빠르게 추린 뒤, long vector로 정밀 재검색하는 구조로 동작한다.
        이 구조는 MRL 방식으로 학습된 임베딩 모델을 전제로 한다.

        Args:
            query_vector (HybridVector): 검색에 사용할 dense 및 sparse 벡터를 포함한 객체.
            limit (int): 최종적으로 반환할 청크의 최대 개수.

        Returns:
            list[Chunk]: RRF 점수 기준으로 정렬된 Chunk 객체 리스트.
                        검색 결과가 없을 경우 빈 리스트([])를 반환한다.
        """
        # 1. MRL Dense 검색 가지 (중첩 구조)
        # 바깥쪽 Prefetch가 최종적으로 RRF에 전달할 정밀 점수를 생산함
        mrl_dense_branch = Prefetch(
            query=query_vector.dense.tolist(), # 정밀 비교용 Long Vector
            using="dense_long",
            limit=limit * 2,
            # 안쪽 Prefetch가 고속으로 후보를 먼저 솎아냄
            prefetch=[
                Prefetch(
                    query=query_vector.dense[:SHORT_DENSE_SIZE].tolist(), # 고속 검색용 Short Vector
                    using="dense_short",
                    limit=limit * 3
                )
            ]
        )

        # 2. Sparse 검색 가지
        sparse_branch = Prefetch(
            query=query_vector.sparse,
            using="sparse",
            limit=limit * 2
        )

        # 3. RRF 알고리즘을 통한 점수 융합
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                mrl_dense_branch,
                sparse_branch
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True
        )

        return [self.payload_to_chunk(point.payload) for point in results.points]
    
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