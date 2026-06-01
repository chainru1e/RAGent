import json
import logging
import time
from typing import Any, Callable, TypeVar

import json_repair

from ragent.config import RERANKING_MODEL
from ragent.models.chunk import Chunk
from ragent.models.vector import HybridVector
from ragent.models.transformed_query import TransformedQuery
from ragent.llm_client import LLMClient
from sentence_transformers import CrossEncoder

logger = logging.getLogger("ragent.retrieval")

T = TypeVar("T")


def _call_with_lock(lock: Any | None, fn: Callable[[], T]) -> tuple[T, float, float]:
    wait_started = time.perf_counter()
    if lock is None:
        run_started = wait_started
        result = fn()
    else:
        with lock:
            run_started = time.perf_counter()
            result = fn()
    finished = time.perf_counter()
    return result, run_started - wait_started, finished - run_started

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
    def __init__(
        self,
        vectordb,
        embedder,
        reranker=None,
        query_transformer=None,
        embedding_lock: Any | None = None,
        rerank_lock: Any | None = None,
    ):
        self.vectordb = vectordb
        self.embedder = embedder
        self.reranker = reranker if reranker is not None else Reranker()
        self.query_transformer = query_transformer if query_transformer is not None else QueryTransformer()
        self.embedding_lock = embedding_lock
        self.rerank_lock = rerank_lock

    def retrieve(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        mode: str = "current_code",
        file_path: str | None = None,
        snapshot_version: int | None = None,
    ) -> list[Chunk]:
        total_started = time.perf_counter()

        # 1. 쿼리 변환
        transform_started = time.perf_counter()
        transformed_queries = self.query_transformer.transform(query)
        transform_ms = (time.perf_counter() - transform_started) * 1000

        # 2. 벡터화 및 HybridVector 조립
        query_vectors = []
        embed_wait_s = 0.0
        embed_s = 0.0
        for transformed_query in transformed_queries:
            def embed_query():
                dense_vec = self.embedder.embed_dense(transformed_query.rewritten)
                keyword_string = " ".join(transformed_query.keywords)
                sparse_vec = self.embedder.embed_sparse(keyword_string)
                return HybridVector(dense=dense_vec, sparse=sparse_vec)

            query_vector, wait_s, run_s = _call_with_lock(
                self.embedding_lock,
                embed_query,
            )
            embed_wait_s += wait_s
            embed_s += run_s
            query_vectors.append(query_vector)

        # 3. 시드 검색
        query_filter = self._build_query_filter(
            workspace_id=workspace_id,
            mode=mode,
            file_path=file_path,
            snapshot_version=snapshot_version,
        )
        qdrant_started = time.perf_counter()
        initial_chunks = self.vectordb.staged_hybrid_search(
            query_vectors,
            query_filter=query_filter,
        )
        qdrant_ms = (time.perf_counter() - qdrant_started) * 1000
        
        if not initial_chunks:
            logger.info(
                "retriever timing: transform_ms=%.1f embed_wait_ms=%.1f embed_ms=%.1f "
                "qdrant_search_ms=%.1f rerank_wait_ms=0.0 rerank_ms=0.0 total_ms=%.1f",
                transform_ms,
                embed_wait_s * 1000,
                embed_s * 1000,
                qdrant_ms,
                (time.perf_counter() - total_started) * 1000,
            )
            return []
        
        # 4. 시드 정제
        reranked_initial_pairs, rerank_wait_s, rerank_s = _call_with_lock(
            self.rerank_lock,
            lambda: self.reranker.rerank(query, initial_chunks),
        )
        static_cutoff_chunks = static_cutoff(reranked_initial_pairs, 0.3)
        dynamic_cutoff_chunks = dynamic_cutoff(static_cutoff_chunks, drop_threshold=0.1, min_chunks=1)
        seed_chunks = [chunk for chunk, score in dynamic_cutoff_chunks]

        logger.info(
            "retriever timing: transform_ms=%.1f embed_wait_ms=%.1f embed_ms=%.1f "
            "qdrant_search_ms=%.1f rerank_wait_ms=%.1f rerank_ms=%.1f total_ms=%.1f",
            transform_ms,
            embed_wait_s * 1000,
            embed_s * 1000,
            qdrant_ms,
            rerank_wait_s * 1000,
            rerank_s * 1000,
            (time.perf_counter() - total_started) * 1000,
        )

        return seed_chunks

    def _build_query_filter(
        self,
        *,
        workspace_id: str | None,
        mode: str,
        file_path: str | None,
        snapshot_version: int | None,
    ):
        if mode == "none":
            return self.vectordb.build_filter(workspace_id="__no_results__")
        if mode == "history":
            return self.vectordb.build_filter(
                workspace_id=workspace_id,
                source_kind="conversation",
            )
        if mode == "historical_code":
            return self.vectordb.build_filter(
                workspace_id=workspace_id,
                source_kind="file_snapshot",
                file_path=file_path,
                is_current=False,
                snapshot_version=snapshot_version,
            )
        return self.vectordb.build_filter(
            workspace_id=workspace_id,
            source_kind="file_snapshot",
            file_path=file_path,
            is_current=True,
            snapshot_version=snapshot_version,
        )
    
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
