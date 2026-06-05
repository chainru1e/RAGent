import logging
from enum import Enum

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    ScrollRequest,
    SparseVectorParams,
    VectorParams,
)

from ragent.config import QDRANT_HOST, QDRANT_PORT, SHORT_DENSE_SIZE, LONG_DENSE_SIZE
from ragent.models.chunk import Chunk, ChunkMetaData
from ragent.models.intent import IntentCategory
from ragent.models.vector import HybridVector

logger = logging.getLogger("ragent.vectordb")


class QdrantStorage:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._init_collection()

    def _init_collection(self):
        try:
            self.client.get_collection(self.collection_name)
            logger.debug("Collection already exists: %s", self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_short": VectorParams(size=SHORT_DENSE_SIZE, distance=Distance.COSINE),
                    "dense_long": VectorParams(size=LONG_DENSE_SIZE, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )
            logger.info("Created collection: %s", self.collection_name)

    # ------------------------------------------------------------------ #
    #  신규: Edit 처리용 메서드                                            #
    # ------------------------------------------------------------------ #

    def get_latest_version(self, file_path: str, func_name: str) -> int | None:
        """
        특정 파일의 특정 함수에 대해 DB에 저장된 현재 최신 버전 번호를 반환한다.
        
        is_latest: True인 청크만 조회하여 version 값을 반환한다.
        DB에 해당 함수가 없으면 None을 반환한다.

        Args:
            file_path (str): 조회할 파일 경로 (예: "auth.py")
            func_name (str): 조회할 함수/클래스 이름 (예: "login")

        Returns:
            int | None: 현재 최신 버전 번호. DB에 없으면 None.
        """
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=file_path)),
                    FieldCondition(key="func_name", match=MatchValue(value=func_name)),
                    FieldCondition(key="is_latest",  match=MatchValue(value=True)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            return None

        return results[0].payload.get("version", 1)

    def mark_outdated(self, file_path: str, func_name: str) -> int:
        """
        특정 파일의 특정 함수에 대해 is_latest: True인 모든 청크를
        is_latest: False로 변경한다. (Soft Delete)

        새 버전 저장 전에 반드시 호출해야 한다.

        Args:
            file_path (str): 대상 파일 경로
            func_name (str): 대상 함수/클래스 이름

        Returns:
            int: 실제로 업데이트된 청크 수
        """
        # is_latest: True인 기존 청크 ID 조회
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=file_path)),
                    FieldCondition(key="func_name", match=MatchValue(value=func_name)),
                    FieldCondition(key="is_latest",  match=MatchValue(value=True)),
                ]
            ),
            limit=100,
            with_payload=False,
            with_vectors=False,
        )

        if not results:
            return 0

        point_ids = [point.id for point in results]

        # is_latest 필드만 False로 덮어씀
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"is_latest": False},
            points=point_ids,
        )

        logger.debug(
            "Marked %d point(s) as outdated — file: %s, func: %s",
            len(point_ids), file_path, func_name
        )
        return len(point_ids)

    # ------------------------------------------------------------------ #
    #  기존 메서드 수정                                                    #
    # ------------------------------------------------------------------ #

    def add_point(self, chunk: Chunk):
        meta = chunk.metadata
        vector = chunk.vector

        point = PointStruct(
            id=meta.chunk_id,
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
                "type": meta.type.value if isinstance(meta.type, Enum) else meta.type,
                "context_prefix": meta.context_prefix,
                # 추가된 3개 필드
                "func_name": meta.func_name,
                "is_latest": meta.is_latest,
                "version": meta.version,
            }
        )

        self.client.upsert(self.collection_name, [point])
        logger.debug("Upserted 1 point to collection %s", self.collection_name)

    def add_points_batch(self, chunks: list[Chunk]) -> int:
        points = []

        for chunk in chunks:
            meta = chunk.metadata
            vector = chunk.vector

            points.append(
                PointStruct(
                    id=meta.chunk_id,
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
                        "type": meta.type.value if isinstance(meta.type, Enum) else meta.type,
                        "context_prefix": meta.context_prefix,
                        # 추가된 3개 필드
                        "func_name": meta.func_name,
                        "is_latest": meta.is_latest,
                        "version": meta.version,
                    }
                )
            )

        self.client.upsert(self.collection_name, points)
        logger.debug("Upserted %d points to collection %s", len(points), self.collection_name)
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
            type=intent_type,
            context_prefix=payload.get("context_prefix"),
            # 추가된 3개 필드 복원
            func_name=payload.get("func_name"),
            is_latest=payload.get("is_latest", True),
            version=payload.get("version", 1),
        )

        return Chunk(
            metadata=metadata,
            payload=payload.get("text"),
            vector=None
        )

    def staged_hybrid_search(self, query_vectors: list[HybridVector], limit: int = 5) -> list[Chunk]:
        """
        여러 개의 서브 쿼리에 대해 2단계 dense 검색과 sparse 검색을 각각 생성하고,
        이 모든 검색 결과를 단 한 번의 RRF 연산으로 융합하는 다중 하이브리드 검색을 수행한다.

        dense 검색은 short vector로 후보군을 빠르게 추린 뒤, long vector로 정밀 재검색하는 구조로 동작한다.
        이 구조는 MRL 방식으로 학습된 임베딩 모델을 전제로 한다.

        is_latest: True 필터를 적용하여 항상 최신 버전 청크만 반환한다.

        Args:
            query_vectors (list[HybridVector]): 검색에 사용할 dense 및 sparse 벡터를 포함한 객체 리스트.
            limit (int): 최종적으로 반환할 청크의 최대 개수.

        Returns:
            list[Chunk]: RRF 점수 기준으로 정렬된 최신 버전 Chunk 객체 리스트.
                        검색 결과가 없을 경우 빈 리스트([])를 반환한다.
        """
        if not query_vectors:
            return []

        prefetch_branches = []

        for query_vector in query_vectors:
            mrl_dense_branch = Prefetch(
                query=query_vector.dense.tolist(),
                using="dense_long",
                limit=limit * 2,
                prefetch=[
                    Prefetch(
                        query=query_vector.dense[:SHORT_DENSE_SIZE].tolist(),
                        using="dense_short",
                        limit=limit * 3
                    )
                ]
            )

            sparse_branch = Prefetch(
                query=query_vector.sparse,
                using="sparse",
                limit=limit * 2
            )

            prefetch_branches.extend([mrl_dense_branch, sparse_branch])

        # 추가: is_latest: True 필터 — 항상 최신 버전만 검색
        latest_filter = Filter(
            must=[
                FieldCondition(key="is_latest", match=MatchValue(value=True))
            ]
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_branches,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=latest_filter,        # 추가
            limit=limit,
            with_payload=True
        )

        chunks = [self.payload_to_chunk(point.payload) for point in results.points]
        logger.debug("Hybrid search returned %d results from collection %s", len(chunks), self.collection_name)
        return chunks

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

    def close(self):
        self.client.close()
