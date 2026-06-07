"""3가지 검색 모드 정의.

Dense Only / Hybrid / Hybrid + Reranker 각각의 검색 함수를 제공한다.
"""

from ragent.models.vector import HybridVector
from ragent.modules.retrieval_modules import Reranker


def retrieve_dense_only(vectordb, embedder, query: str, k: int = 5) -> list:
    """Dense 벡터만 사용하는 단순 검색."""
    dense_vec = embedder.embed_dense(query)
    return vectordb.dense_only_search(dense_vec, limit=k)


def retrieve_hybrid(vectordb, embedder, query: str, k: int = 5) -> list:
    """Dense + Sparse BM25 RRF 융합 검색 (Reranker 없음)."""
    dense_vec = embedder.embed_dense(query)
    sparse_vec = embedder.embed_sparse(query)
    hybrid_vec = HybridVector(dense=dense_vec, sparse=sparse_vec)
    return vectordb.staged_hybrid_search([hybrid_vec], limit=k)


def retrieve_hybrid_reranker(vectordb, embedder, query: str, k: int = 5) -> list:
    """Hybrid 검색 후 Cross-Encoder Reranker로 재순위화."""
    chunks = retrieve_hybrid(vectordb, embedder, query, k=k * 2)
    if not chunks:
        return []

    reranker = Reranker()
    scored = reranker.rerank(query, chunks)
    return [chunk for chunk, _ in scored[:k]]


MODES = {
    "Dense Only": retrieve_dense_only,
    "Hybrid": retrieve_hybrid,
    "Hybrid + Reranker": retrieve_hybrid_reranker,
}
