from ragent.config import RERANKING_MODEL
from ragent.models.chunk import Chunk
from sentence_transformers import CrossEncoder

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
    def __init__(self, vectordb, embedder, reranker=None):
        self.vectordb = vectordb
        self.embedder = embedder
        self.reranker = reranker if reranker is not None else Reranker()

    def retrieve(self, query: str) -> list[Chunk]:
        # 1. 초기 시드 검색
        query_vector = self.embedder.embed(query)
        initial_chunks = self.vectordb.staged_hybrid_search(query_vector=query_vector)
        
        if not initial_chunks:
            return []
        
        # 2. 시드 정제
        reranked_initial_pairs = self.reranker.rerank(query, initial_chunks)
        seed_chunks = cutoff(reranked_initial_pairs, drop_threshold=0.1, min_chunks=1)

        return seed_chunks
    
class Reranker:
    """검색된 청크들을 Cross-Encoder 모델을 이용해 재평가하고 정렬하는 클래스"""
    def __init__(self):
        self.model = CrossEncoder(RERANKING_MODEL)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        """
        주어진 쿼리를 기준으로 각 청크의 관련성 점수를 산출하고, 가장 연관성이 높은 순서대로 청크를 정렬한다.

        Args:
            query (str): 평가의 기준이 되는 대상 쿼리 또는 문장.
            chunks (list[Chunk]): 유사도를 평가할 대상 청크 객체 리스트.

        Returns:
            list[tuple[Chunk, float]]: (Chunk, 점수) 형태의 튜플 리스트. 
                                       점수가 높은 순으로 내림차순 정렬되어 반환된다.
        """
        if not chunks:
            return []

        pairs = [[query, chunk.payload or ""] for chunk in chunks]
        
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            return [(chunk, 0.0) for chunk in chunks]

        scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return scored_chunks
