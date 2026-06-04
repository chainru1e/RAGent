import json
import logging

import json_repair

from ragent.config import RERANKING_MODEL, RERANK_BATCH_SIZE, RERANK_MAX_LENGTH
from ragent.models.chunk import Chunk
from ragent.models.vector import HybridVector
from ragent.models.transformed_query import TransformedQuery
from ragent.llm_client import LLMClient
from sentence_transformers import CrossEncoder

logger = logging.getLogger("ragent")

# rerank() 가 predict 2차 재시도까지 실패했을 때 쓰는 pass-through 점수.
# static_cutoff(0.3) 문턱 이상의 균일값이라, 전 청크가 cutoff 에 전멸하지 않고
# (= reranking 생략, hybrid 후보 순서 유지) dynamic_cutoff 도 낙폭 0 이라 통과한다.
# 0.0 으로 삼키지 않기 위한 degrade 경로이지 정상 점수가 아니다.
RERANK_FALLBACK_SCORE = 0.5

def static_cutoff(scored_chunks: list[tuple[Chunk, float]], threshold: float) -> list[tuple[Chunk, float]]:
    """
    주어진 임계값(threshold) 이상의 점수를 가진 청크들만 필터링하여 반환한다.
    
    Args:
        scored_chunks: (Chunk, 점수) 형태의 튜플 리스트.
        threshold: 통과시키기 위한 최소 점수 기준.
        
    Returns:
        list[tuple[Chunk, float]]: 정적 컷오프 조건을 통과한 (Chunk, 점수) 형태의 튜플 리스트.
    """
    if not scored_chunks:
        return []
    
    return [(chunk, score) for chunk, score in scored_chunks if score >= threshold]

def dynamic_cutoff(scored_chunks: list[tuple[Chunk, float]], drop_threshold: float = 0.1, min_chunks: int = 1) -> list[tuple[Chunk, float]]:
    """
    청크들의 유사도 점수 낙폭을 분석하여 연관성이 떨어지는 하위 청크들을 잘라낸다.
    입력된 데이터는 내부적으로 점수 기준 내림차순 정렬을 적용한 뒤 컷오프를 수행한다.
    
    Args:
        scored_chunks: (Chunk, 점수) 형태의 튜플 리스트.
        drop_threshold: 이전 청크 대비 점수가 이 값보다 크게 떨어지면 컷오프를 실행한다. 기본값 0.1.
        min_chunks: 점수 낙폭이 크더라도 무조건 결과에 포함시킬 최소 청크 개수. 기본값 1.
        
    Returns:
        list[tuple[Chunk, float]]: 동적 컷오프 조건을 통과하여 살아남은 (Chunk, 점수) 형태의 튜플 리스트.
    """
    if not scored_chunks:
        return []
    
    sorted_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
    
    if len(sorted_chunks) <= min_chunks:
        return sorted_chunks

    filtered_chunks = [sorted_chunks[0]]

    drop_detected = False
    for i in range(1, len(sorted_chunks)):
        current_score = sorted_chunks[i][1]
        prev_score = sorted_chunks[i-1][1]
        
        drop = prev_score - current_score
        
        if drop > drop_threshold:
            drop_detected = True
        
        if drop_detected and len(filtered_chunks) >= min_chunks:
            break
            
        filtered_chunks.append(sorted_chunks[i])

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
        static_cutoff_chunks = static_cutoff(reranked_initial_pairs, 0.3)
        dynamic_cutoff_chunks = dynamic_cutoff(static_cutoff_chunks, drop_threshold=0.1, min_chunks=1)
        seed_chunks = [chunk for chunk, score in dynamic_cutoff_chunks]

        return seed_chunks
    
class Reranker:
    """검색된 청크들을 Cross-Encoder 모델을 이용해 재평가하고 정렬하는 클래스"""
    def __init__(self):
        self.model = CrossEncoder(RERANKING_MODEL)
        # 캡이 설정된 경우에만 적용한다(None 이면 모델 기본 max_length 유지).
        if RERANK_MAX_LENGTH is not None:
            self.model.max_length = RERANK_MAX_LENGTH

    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        """
        주어진 쿼리를 기준으로 각 청크의 관련성 점수를 산출하고, 가장 연관성이 높은 순서대로 청크를 정렬한다.

        Args:
            query (str): 평가의 기준이 되는 대상 쿼리 또는 문장.
            chunks (list[Chunk]): 유사도를 평가할 대상 청크 객체 리스트.

        Returns:
            list[tuple[Chunk, float]]: (Chunk, 점수) 형태의 튜플 리스트.
                                       점수가 높은 순으로 내림차순 정렬되어 반환된다.

        장애 처리:
            predict 가 긴 청크에서 MPS OOM 등으로 예외를 내면 batch_size=1 로 1회
            재시도한다. 재시도까지 실패하면 logger.exception 으로 원인을 남기고,
            전 청크에 0.0 을 반환하는 대신 **입력 순서를 보존한 균일 fallback 점수**
            (RERANK_FALLBACK_SCORE, 문턱 0.3 이상)로 degrade 한다. 그래야
            static_cutoff/dynamic_cutoff 가 전 청크를 전멸시키지 않고 hybrid 후보가
            그대로 유지된다(= reranking 생략). 어떤 경로에서도 전 청크 0.0 은 반환하지 않는다.
        """
        if not chunks:
            return []

        pairs = [[query, chunk.payload or ""] for chunk in chunks]

        try:
            scores = self.model.predict(pairs, batch_size=RERANK_BATCH_SIZE)
        except Exception:
            logger.exception(
                "reranker: predict failed at batch_size=%s; retrying with batch_size=1",
                RERANK_BATCH_SIZE,
            )
            try:
                scores = self.model.predict(pairs, batch_size=1)
            except Exception:
                logger.exception(
                    "reranker: predict failed at batch_size=1; "
                    "degrading to pass-through (uniform fallback score, no rerank)"
                )
                # 입력 순서 보존 + 균일 점수(≥0.3): cutoff 가 전멸시키지 않음.
                return [(chunk, RERANK_FALLBACK_SCORE) for chunk in chunks]

        scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return scored_chunks

class QueryTransformer:
    def __init__(self):
        system_prompt = """
            You are an expert in query transformation for Retrieval-Augmented Generation (RAG) and vector database search optimization.
            Your task is to analyze the user's raw input query and transform it into an optimized JSON format.

            ### CRITICAL LANGUAGE RULE:
            You MUST output the `rewritten` and `keywords` fields in the EXACT SAME LANGUAGE as the user's raw input.

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
                parsed_data = json_repair.loads(response_text)
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