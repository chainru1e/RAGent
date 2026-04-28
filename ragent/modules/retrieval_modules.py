from ragent.models.chunk import Chunk
from qdrant_client.models import Filter, FieldCondition, MatchValue

class Retriever:
    def __init__(self, vectordb, embedder):
        self.vectordb = vectordb
        self.embedder = embedder

    def retrieve(self, query: str) -> list[Chunk]:
        query_vector = self.embedder.embed(query)
        search_results = self.vectordb.hybrid_search(query_vector=query_vector)
        
        return search_results
    
class MetadataExpander:
    """검색된 청크의 메타데이터를 기반으로 추가적인 청크를 검색하는 클래스"""
    def __init__(self, vectordb):
        self.vectordb = vectordb

    def _fetch_by_filter(self, conditions: dict, limit: int) -> list[Chunk]:
        """
        주어진 다중 메타데이터 조건을 모두 만족하는 청크들을 조회한다.
        
        Args:
            conditions (dict): 검색에 사용할 메타데이터 키-값 쌍.
                               값이 None인 키는 필터링 조건에서 자동으로 제외한다.
            limit (int): 데이터베이스에서 스크롤하여 가져올 청크의 최대 개수.
            
        Returns:
            list[Chunk]: 조건을 만족하며 역직렬화가 완료된 순수 Chunk 객체 리스트. 
                         검색 결과가 없거나 통신 중 예외가 발생할 경우 빈 리스트([])를 반환한다.
        """

        # Qdrant 필터 생성
        must_conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in conditions.items() if v is not None
        ]
        qdrant_filter = Filter(must=must_conditions)

        # 메타데이터 기반 검색
        try:
            records, _ = self.vectordb.client.scroll(
                collection_name=self.vectordb.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True
            )
            return [self.vectordb.payload_to_chunk(record.payload) for record in records]
            
        except Exception:
            return []

    def expand_to_parent(self, chunk: Chunk) -> list[Chunk]:
        """코드 -> 대화 텍스트"""
        if not chunk.metadata.parent_id:
            return []
        return self._fetch_by_filter(conditions={"chunk_id": chunk.metadata.parent_id}, limit=1)

    def expand_to_children(self, chunk: Chunk) -> list[Chunk]:
        """대화 텍스트 -> 소속된 코드들"""
        if not chunk.metadata.chunk_id:
            return []
        return self._fetch_by_filter(conditions={"parent_id": chunk.metadata.chunk_id}, limit=50)
    
    def expand_to_siblings(self, chunk: Chunk) -> list[Chunk]:
        """코드 -> 같은 대화를 공유하는 다른 코드들"""
        siblings = self._fetch_by_filter(conditions={"parent_id": chunk.metadata.parent_id}, limit=50)
        # 자기 자신 제외
        return [sib for sib in siblings if sib.metadata.chunk_id != chunk.metadata.chunk_id]
    
    def expand_same_file(self, chunk: Chunk) -> list[Chunk]:
        """코드 -> 같은 파일의 전체 코드"""
        if not chunk.metadata.file_path or not chunk.metadata.parent_id:
            return []
        return self._fetch_by_filter(
            conditions={
                "file_path": chunk.metadata.file_path,
                "parent_id": chunk.metadata.parent_id
            }, 
            limit=50
        )