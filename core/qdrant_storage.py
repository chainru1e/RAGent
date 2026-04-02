# ════════════════════════════════════════════════════════════════════════════
# core/qdrant_storage.py - Qdrant 벡터 저장소
# ════════════════════════════════════════════════════════════════════════════

import numpy as np
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    MatchValue
)
from config.settings import QDRANT_COLLECTION, QDRANT_PATH


class QdrantStorage:
    '''
    Qdrant 벡터 저장소
    
    저장 구조:
    - id: 정수 (chunk_id hash)
    - vector: [384개의 부동소수점]
    - payload: {chunk_id, text, metadata...}
    '''
    
    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        vector_size: int = 384,
        path: str = QDRANT_PATH
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        print(f"🔧 Qdrant 저장소 초기화")
        print(f"   컬렉션: {collection_name}")
        print(f"   벡터 차원: {vector_size}")
        print(f"   경로: {path}")
        
        self.client = QdrantClient(path=path)
        
        try:
            self.client.get_collection(collection_name)
            print(f"✅ 기존 컬렉션 로드: {collection_name}")
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ 새 컬렉션 생성: {collection_name}")
    
    def add_point(
        self,
        chunk_id: str,
        vector: np.ndarray,
        text: str,
        metadata: Dict = None
    ) -> bool:
        '''단일 포인트 저장'''
        
        try:
            point = PointStruct(
                id=self._string_to_id(chunk_id),
                vector=vector.tolist(),
                payload={
                    "chunk_id": chunk_id,
                    "text": text,
                    **(metadata or {})
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return True
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
            return False
    
    def add_points_batch(self, chunks: List[Dict]) -> int:
        '''배치 저장 (권장)'''
        
        try:
            points = [
                PointStruct(
                    id=self._string_to_id(chunk["chunk_id"]),
                    vector=chunk["vector"].tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        **(chunk.get("metadata", {}))
                    }
                )
                for chunk in chunks
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"✅ 배치 저장: {len(chunks)}개")
            return len(chunks)
        
        except Exception as e:
            print(f"❌ 배치 저장 실패: {e}")
            return 0
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_dict: Dict = None
    ) -> List[Tuple[str, float, str]]:
        '''벡터 검색'''
        
        try:
            query_filter = None
            if filter_dict:
                conditions = [
                    MatchValue(key=k, value=v)
                    for k, v in filter_dict.items()
                ]
                query_filter = Filter(must=conditions)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=query_filter,
                limit=top_k
            )
            
            return [
                (result.payload["chunk_id"], result.score, result.payload["text"])
                for result in results
            ]
        
        except Exception as e:
            print(f"❌ 검색 실패: {e}")
            return []
    
    def get_stats(self) -> Dict:
        '''저장소 통계 조회'''
        
        try:
            info = self.client.get_collection(self.collection_name)
            
            return {
                "collection": self.collection_name,
                "total_points": info.points_count,
                "vector_size": self.vector_size,
                "status": str(info.status)
            }
        
        except Exception as e:
            print(f"❌ 통계 조회 실패: {e}")
            return {}
    
    def _string_to_id(self, s: str) -> int:
        '''문자열 ID를 정수로 변환'''
        return int(hash(s) & 0x7FFFFFFF)
