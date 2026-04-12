from ragent.config import QDRANT_DIR, DENSE_SIZE, ensure_dirs
from enum import Enum
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    SparseVectorParams,
    Distance
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