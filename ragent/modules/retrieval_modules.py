from ragent.models.chunk import Chunk
from qdrant_client.models import Filter, FieldCondition, MatchValue

def cutoff(scored_chunks: list[tuple[Chunk, float]], drop_threshold: float = 0.1, min_chunks: int = 1) -> list[Chunk]:
    """
    청크들의 유사도 점수 낙폭을 분석하여 연관성이 떨어지는 하위 청크들을 잘라낸다.
    입력된 데이터는 내부적으로 점수 기준 내림차순 정렬을 적용한 뒤 컷오프를 수행한다.
    
    Args:
        scored_chunks: (Chunk, 점수) 형태의 튜플 리스트.
        drop_threshold: 이전 청크 대비 점수가 이 값보다 크게 떨어지면 컷오프를 실행한다. 기본값 0.1.
        min_chunks: 점수 낙폭이 크더라도 무조건 결과에 포함시킬 최소 청크 개수. 기본값 1.
        
    Returns:
        동적 컷오프 조건을 통과하여 살아남은 순수 Chunk 객체 리스트.
    """
    if not scored_chunks:
        return []
    
    sorted_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
    
    if len(sorted_chunks) <= min_chunks:
        return [chunk for chunk, score in sorted_chunks]

    filtered_chunks = [sorted_chunks[0][0]]

    drop_detected = False
    for i in range(1, len(sorted_chunks)):
        current_score = sorted_chunks[i][1]
        prev_score = sorted_chunks[i-1][1]
        
        drop = prev_score - current_score
        
        if drop > drop_threshold:
            drop_detected = True
        
        if drop_detected and len(filtered_chunks) >= min_chunks:
            break
            
        filtered_chunks.append(sorted_chunks[i][0])

    return filtered_chunks

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