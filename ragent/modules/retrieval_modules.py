import json

from ragent.config import RERANKING_MODEL
from ragent.models.chunk import Chunk
from ragent.models.vector import HybridVector
from ragent.models.transformed_query import TransformedQuery
from ragent.llm_client import LLMClient
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
    def __init__(self, vectordb, embedder, reranker=None, query_transformer=None):
        self.vectordb = vectordb
        self.embedder = embedder
        self.reranker = reranker if reranker is not None else Reranker()
        self.query_transformer = query_transformer if query_transformer is not None else QueryTransformer()

    def retrieve(self, query: str) -> list[Chunk]:
        # 1. 쿼리 변환
        transformed_queries = self.query_transformer.transform(query)

        # 2. 벡터화 및 HybridVector 조립
        query_vectors = []
        for transformed_query in transformed_queries:
            dense_vec = self.embedder.embed_dense(transformed_query.rewritten)
            keyword_string = " ".join(transformed_query.keywords)
            sparse_vec = self.embedder.embed_sparse(keyword_string)
            query_vectors.append(HybridVector(dense=dense_vec, sparse=sparse_vec))

        # 3. 시드 검색
        initial_chunks = self.vectordb.staged_hybrid_search(query_vectors)
        
        if not initial_chunks:
            return []
        
        # 4. 시드 정제
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

class QueryTransformer:
    def __init__(self):
        system_prompt = """
            You are an expert in query transformation for Retrieval-Augmented Generation (RAG) and vector database search optimization.
            Your task is to analyze the user's raw input query and transform it into an optimized JSON format.

            ### Instructions:
            1. **Decomposition:** Analyze if the user's input contains multiple distinct questions or topics. If so, break it down into separate, independent sub-queries. If it is a single topic, keep it as one query.
            2. **Rewriting:** For each resulting query, remove unnecessary conversational filler, emotional language, complaints, and ambiguous pronouns. Rewrite it into a clear, concise, and explicit search objective in the language of the original user query.
            3. **Expansion:** For each rewritten query, generate a list of highly relevant technical keywords, synonyms, and related domain concepts. These keywords will be used for Sparse/BM25 retrieval, so focus on exact and specific terminology in the language of the original user query.

            ### Output Constraint:
            You must respond STRICTLY with a valid JSON object matching the exact schema below. Do not include any markdown formatting, explanations, or conversational text outside the JSON.

            ### Output Schema:
            {
            "queries": [
                {
                "rewritten": "[A clear, standalone search query]",
                "keywords": ["[keyword1]", "[keyword2]", "[keyword3]"]
                }
            ]
            }
        """
        self.llm_client = LLMClient(system_prompt=system_prompt)

    def transform(self, query: str) -> list[TransformedQuery]:
            """
            사용자 질문을 분석하여 재작성된 쿼리와 키워드 리스트를 반환합니다.

            Args:
                query (str): 사용자 질문
            
            Returns:
                list[TransformedQuery]: 재작성된 쿼리와 키워드를 담은 객체 리스트
            """
            try:
                response_text = self.llm_client.ask(query, temperature=0.0)
                parsed_data = json.loads(response_text)
                queries = parsed_data.get("queries", [])
                if not queries:
                    return [TransformedQuery(rewritten=query, keywords=[])]
                
                result = []
                for q in queries:
                    result.append(
                        TransformedQuery(
                            rewritten=q.get("rewritten", query),
                            keywords=q.get("keywords", [])
                        )
                    )
                return result
            except json.JSONDecodeError as e:
                return [TransformedQuery(rewritten=query, keywords=[])]
            except Exception as e:
                return [TransformedQuery(rewritten=query, keywords=[])]