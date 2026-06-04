"""ranx Run 생성기 — 기존 RAGent 검색 코드를 재사용해 순위 리스트를 만든다.

설계 요약
=========
- 평가 대상은 **Reranker.rerank 직후**의 순위 리스트다. 현재 코드베이스에는
  MetadataExpander 가 없으므로(조사 결과) retrieve() 출력이 곧 "확장 이전" 순위
  리스트이며, 여기서는 cutoff 이전·rerank 직후의 top-N 순위를 평가 단위로 본다.
- 점수 확보: QdrantStorage.client (raw QdrantClient) 로 **read-only** query_points
  를 모드별로 직접 구성한다. 기존 staged_hybrid_search 는 (1) hybrid 전용이고
  (2) payload_to_chunk 가 point.score 를 버려 점수를 돌려주지 않으므로, 점수가
  필요한 평가에는 부적합하다. hybrid 모드는 staged_hybrid_search 의 RRF 구조
  (FusionQuery RRF, dense 2단(short→long) + sparse, 동일 limit 배수)를 그대로
  미러링한다. → verify_mirror_fidelity() 로 실제 일치 여부를 검증한다.

⚠️ production 과의 의도된 차이 3가지
-----------------------------------
1. reranker 풀 크기: production(retrieve)은 staged_hybrid_search(limit=5)의 5개만
   reranker 에 넣는다. 그러면 rerank on/off 가 같은 5개를 순서만 바꾸는 꼴이라
   P@5·R@5·Hit@5 가 동일해져 ablation 이 무의미하다. 따라서 평가 러너는 depth=N
   (기본 20)로 검색한 **N개 전부를 reranker 가 재정렬**한 뒤 @5 를 평가한다.
   - rerank off = hybrid RRF(또는 단일 유사도) 순서의 top-k
   - rerank on  = reranker 가 N개를 재정렬한 순서의 top-k
2. @k / depth: 고정 주입 컷오프 M 은 존재하지 않는다(static_cutoff(0.3) +
   dynamic_cutoff(0.1) 로 가변). 평가 @k 는 후보 풀/주입 상한인 limit=5 에서 따온
   k=5 를 기본값으로 쓰고, depth=N=20 로 넉넉히 가져와 R@20(천장)도 함께 본다.
3. 쿼리 변환: production 은 QueryTransformer(LLM 재작성)를 **항상 ON** 으로 쓴다.
   평가 러너는 재현성을 위해 기본 OFF(원본 쿼리를 그대로 임베딩)이며, config 의
   transform=True 로 production 충실 모드를 켤 수 있다(LLM 서버 필요·비결정적).

이 모듈은 기존 코드를 import 만 하고 절대 수정하지 않으며, Qdrant 에 대해
query_points/get_collection/scroll/retrieve 같은 read-only 호출만 수행한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch

from ragent.config import QDRANT_HOST, QDRANT_PORT, SHORT_DENSE_SIZE
from ragent.models.chunk import Chunk, ChunkMetaData
from ragent.models.vector import HybridVector
from ragent.modules.retrieval_modules import QueryTransformer, Reranker
from ragent.vectordb_client import QdrantStorage

logger = logging.getLogger("ragent.evaluation.runner")

# staged_hybrid_search(vectordb_client.py) 의 limit 배수를 그대로 미러링한다.
# dense outer(long) = limit*2, dense inner(short) = limit*3, sparse = limit*2.
DENSE_OUTER_MULT = 2
DENSE_INNER_MULT = 3
SPARSE_MULT = 2

VALID_MODES = ("dense", "sparse", "hybrid")
VALID_DENSE_VARIANTS = ("mrl_2stage", "long_only")

# production(retrieval_modules.retrieve) 이 reranker 점수에 적용하는 주입 컷오프
# 상수. 평가 진단(cutoff diagnostics)에서 동일 값으로 재현한다.
STATIC_CUTOFF_THRESHOLD = 0.3
DYNAMIC_CUTOFF_DROP = 0.1
DYNAMIC_CUTOFF_MIN_CHUNKS = 1


@dataclass
class RetrievalConfig:
    """단일 ablation 셀의 검색 설정."""

    mode: str  # "dense" | "sparse" | "hybrid"
    rerank: bool
    collection: str
    dense_variant: str = "mrl_2stage"  # "mrl_2stage" | "long_only" (dense/hybrid 만)
    transform: bool = False  # QueryTransformer(LLM) on/off. 기본 OFF(재현성).
    name: str = ""

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.dense_variant not in VALID_DENSE_VARIANTS:
            raise ValueError(
                f"dense_variant must be one of {VALID_DENSE_VARIANTS}, "
                f"got {self.dense_variant!r}"
            )
        if not self.collection:
            raise ValueError("collection must be a non-empty string")
        if not self.name:
            self.name = self.default_name()

    def default_name(self) -> str:
        """사람이 읽기 쉬운 run 이름. ranx Report 의 모델명으로 쓰인다."""
        if self.mode == "dense":
            base = f"dense_{self.dense_variant}"
        else:
            base = self.mode
        rr = "rerank=on" if self.rerank else "rerank=off"
        tr = "+transform" if self.transform else ""
        return f"{base}|{rr}{tr}"


@dataclass
class RunResult:
    """run_config 의 결과 묶음."""

    name: str
    config: RetrievalConfig
    run: dict[str, dict[str, float]]  # ranx Run dict: {query_id: {chunk_id: score}}
    skipped: list[str]  # retrieval 실패로 스킵된 query_id
    empty: list[str]  # 검색 결과가 0건인 query_id


class EvalRetriever:
    """모드별 read-only 검색 + (옵션) reranking 으로 ranx Run 을 만든다.

    무거운 자원(임베더 GGUF, reranker CrossEncoder)은 lazy 로 한 번만 로드해
    여러 config 에 재사용한다.
    """

    def __init__(self, client: QdrantClient | None = None):
        self.client = client or QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._embedder = None
        self._reranker = None
        self._transformer = None

    # ---- lazy 자원 ---------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            # 지연 import: HybridEmbedding 생성 시 GGUF 다운로드/로딩이 일어난다.
            from ragent.modules.embedding_modules import HybridEmbedding

            logger.info("Loading HybridEmbedding (dense GGUF + bm25)...")
            self._embedder = HybridEmbedding()
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            logger.info("Loading Reranker (CrossEncoder)...")
            self._reranker = Reranker()
        return self._reranker

    @property
    def transformer(self) -> QueryTransformer:
        if self._transformer is None:
            logger.info("Loading QueryTransformer (LLM client)...")
            self._transformer = QueryTransformer()
        return self._transformer

    # ---- 컬렉션 검증 -------------------------------------------------------
    def ensure_collection(self, collection: str) -> None:
        """컬렉션 존재를 확인한다. 없으면 loud fail (절대 생성하지 않는다)."""
        try:
            self.client.get_collection(collection)
        except Exception as exc:  # noqa: BLE001 - 원인 그대로 노출
            raise RuntimeError(
                f"Qdrant collection not found or unreachable: {collection!r}. "
                f"cause: {type(exc).__name__}: {exc}"
            ) from exc

    def chunk_ids_exist(self, collection: str, chunk_ids: list[str]) -> set[str]:
        """주어진 chunk_id 중 컬렉션에 실재하는 것들의 집합 (read-only)."""
        if not chunk_ids:
            return set()
        found = self.client.retrieve(
            collection_name=collection,
            ids=chunk_ids,
            with_payload=False,
            with_vectors=False,
        )
        return {str(p.id) for p in found}

    # ---- 쿼리 임베딩 ------------------------------------------------------
    def _embed_query(
        self, query: str, transform: bool
    ) -> list[tuple[np.ndarray, object]]:
        """쿼리를 (dense ndarray, sparse SparseVector) 튜플 리스트로 임베딩한다.

        transform=False: 원본 쿼리 1개. dense/sparse 모두 원본 텍스트를 쓴다.
        transform=True:  QueryTransformer 가 만든 서브쿼리마다 1개씩. dense 는
                         rewritten, sparse 는 keywords(공백조인)를 쓴다 —
                         production(retrieve) 의 입력 구성과 동일하게 맞춘다.
        """
        if transform:
            tqs = self.transformer.transform(query)
            vecs = []
            for tq in tqs:
                dense = self.embedder.embed_dense(tq.rewritten)
                sparse = self.embedder.embed_sparse(" ".join(tq.keywords))
                vecs.append((dense, sparse))
            if vecs:
                return vecs
            # 변환이 빈 결과면 원본으로 폴백(production 동작과 일치)
        dense = self.embedder.embed_dense(query)
        sparse = self.embedder.embed_sparse(query)
        return [(dense, sparse)]

    # ---- 브랜치 빌더 ------------------------------------------------------
    def _dense_branch(
        self, dense: np.ndarray, dense_variant: str, depth: int
    ) -> Prefetch:
        d_long = dense.tolist()
        if dense_variant == "long_only":
            return Prefetch(
                query=d_long, using="dense_long", limit=depth * DENSE_OUTER_MULT
            )
        d_short = dense[:SHORT_DENSE_SIZE].tolist()
        # MRL 2단: short 로 후보를 빠르게 추린 뒤 long 으로 정밀 재검색
        return Prefetch(
            query=d_long,
            using="dense_long",
            limit=depth * DENSE_OUTER_MULT,
            prefetch=[
                Prefetch(
                    query=d_short,
                    using="dense_short",
                    limit=depth * DENSE_INNER_MULT,
                )
            ],
        )

    def _sparse_branch(self, sparse: object, depth: int) -> Prefetch:
        return Prefetch(query=sparse, using="sparse", limit=depth * SPARSE_MULT)

    # ---- 모드별 검색 ------------------------------------------------------
    def _raw_search(self, config: RetrievalConfig, vecs, depth: int):
        """모드/변형/서브쿼리 수에 따라 query_points 를 구성해 호출한다.

        서브쿼리가 1개(transform off 의 dense/sparse)면 native 유사도 점수를
        보존하기 위해 fusion 없이 단일 벡터로 질의한다. 서브쿼리가 여러 개거나
        hybrid 모드면 RRF 로 융합한다(점수는 RRF 점수가 된다).
        """
        coll = config.collection
        single = len(vecs) == 1

        if config.mode == "dense":
            if single:
                dense = vecs[0][0]
                if config.dense_variant == "long_only":
                    return self.client.query_points(
                        coll,
                        query=dense.tolist(),
                        using="dense_long",
                        limit=depth,
                        with_payload=True,
                    )
                return self.client.query_points(
                    coll,
                    prefetch=[
                        Prefetch(
                            query=dense[:SHORT_DENSE_SIZE].tolist(),
                            using="dense_short",
                            limit=depth * DENSE_INNER_MULT,
                        )
                    ],
                    query=dense.tolist(),
                    using="dense_long",
                    limit=depth,
                    with_payload=True,
                )
            branches = [
                self._dense_branch(d, config.dense_variant, depth) for d, _ in vecs
            ]
        elif config.mode == "sparse":
            if single:
                return self.client.query_points(
                    coll,
                    query=vecs[0][1],
                    using="sparse",
                    limit=depth,
                    with_payload=True,
                )
            branches = [self._sparse_branch(s, depth) for _, s in vecs]
        else:  # hybrid — staged_hybrid_search 미러링
            branches = []
            for dense, sparse in vecs:
                branches.append(self._dense_branch(dense, config.dense_variant, depth))
                branches.append(self._sparse_branch(sparse, depth))

        return self.client.query_points(
            coll,
            prefetch=branches,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=depth,
            with_payload=True,
        )

    # ---- reranking --------------------------------------------------------
    def _rerank(self, query: str, points) -> list[tuple[str, float]]:
        """검색된 point 들을 Reranker(CrossEncoder)로 전부 재정렬한다.

        production(retrieve)과 동일하게 **원본 쿼리** 를 reranker 입력으로 쓴다
        (transform 된 쿼리가 아님). reranker 는 chunk.payload(=본문 text)를 본다.
        """
        chunks = [
            Chunk(
                metadata=ChunkMetaData(chunk_id=str(p.payload.get("chunk_id"))),
                payload=p.payload.get("text") or "",
            )
            for p in points
        ]
        scored = self.reranker.rerank(query, chunks)  # [(Chunk, score)] desc
        return [(c.metadata.chunk_id, float(s)) for c, s in scored]

    # ---- 단일 쿼리 검색 ---------------------------------------------------
    def search_one(
        self, query: str, config: RetrievalConfig, depth: int
    ) -> list[tuple[str, float]]:
        """한 쿼리에 대한 (chunk_id, score) 순위 리스트(top-depth)를 반환한다."""
        vecs = self._embed_query(query, config.transform)
        res = self._raw_search(config, vecs, depth)
        if config.rerank:
            return self._rerank(query, res.points)
        return [
            (str(p.payload.get("chunk_id")), float(p.score)) for p in res.points
        ]

    # ---- config 단위 Run 생성 --------------------------------------------
    def run_config(
        self, config: RetrievalConfig, queries: dict[str, str], depth: int = 20
    ) -> RunResult:
        """ablation config 하나로 전체 쿼리를 돌려 ranx Run dict 를 만든다.

        개별 쿼리 retrieval 실패는 로그+스킵+카운트하고 전체를 멈추지 않는다.
        검색 결과가 0건인 쿼리는 빈 dict 로 두고 empty 에 기록한다.
        """
        self.ensure_collection(config.collection)
        run: dict[str, dict[str, float]] = {}
        skipped: list[str] = []
        empty: list[str] = []

        for qid, query in queries.items():
            try:
                pairs = self.search_one(query, config, depth)
            except Exception as exc:  # noqa: BLE001 - 쿼리 단위 격리
                logger.warning(
                    "[%s] retrieval failed for query_id=%s: %s: %s",
                    config.name, qid, type(exc).__name__, exc,
                )
                skipped.append(qid)
                continue

            if not pairs:
                empty.append(qid)
            # 동일 chunk_id 가 여러 서브쿼리 브랜치에서 중복되면 더 높은 점수 유지
            scores: dict[str, float] = {}
            for cid, score in pairs:
                if cid not in scores or score > scores[cid]:
                    scores[cid] = score
            run[qid] = scores

        return RunResult(
            name=config.name, config=config, run=run, skipped=skipped, empty=empty
        )

    # ---- 미러 충실도 검증 -------------------------------------------------
    def verify_mirror_fidelity(
        self, queries: dict[str, str], collection: str, sample: int = 3
    ) -> list[dict]:
        """hybrid 미러가 production staged_hybrid_search 와 같은 결과를 내는지 검증.

        production 경로(QdrantStorage.staged_hybrid_search, limit=5)와 본 러너의
        hybrid 미러(mrl_2stage, depth=5, transform off)를 같은 쿼리·같은 임베딩으로
        비교해 top-5 chunk_id 순서가 일치하는지 본다.

        중요(1): dense 임베딩(llama.cpp)은 호출마다 미세하게 달라져 RRF 동점 구간의
        순서를 흔든다. 따라서 두 경로에 **동일한** dense/sparse 벡터를 주입해야
        구조(미러링) 자체만 비교된다. 미러 경로도 search_one(재임베딩) 대신
        _raw_search 에 같은 벡터를 직접 넘긴다.

        중요(2): 동일 벡터를 줘도 Qdrant 의 근사 검색/세그먼트 병합은 RRF 동점
        구간의 순서를 호출마다 바꾼다(production staged_hybrid_search 를 같은 벡터로
        3회 호출해도 top-5 순서가 흔들림 — 단, chunk_id 집합은 동일). 그러므로
        충실도 기준은 "top-5 chunk_id **집합** 일치"(set_match)이고, 정확한 순서
        일치(order_match)는 동점 비결정성 때문에 참고용으로만 본다.

        주: 기존 컬렉션을 대상으로만 호출하므로 QdrantStorage(collection) 의
        _init_collection 은 기존 컬렉션을 찾고 아무것도 생성하지 않는다(read-only).
        """
        self.ensure_collection(collection)
        storage = QdrantStorage(collection)  # 기존 컬렉션 → 생성 부작용 없음
        report: list[dict] = []
        for qid, query in list(queries.items())[:sample]:
            dense = self.embedder.embed_dense(query)
            sparse = self.embedder.embed_sparse(query)
            prod_chunks = storage.staged_hybrid_search(
                [HybridVector(dense=dense, sparse=sparse)], limit=5
            )
            prod_ids = [c.metadata.chunk_id for c in prod_chunks]

            mirror_cfg = RetrievalConfig(
                mode="hybrid",
                rerank=False,
                collection=collection,
                dense_variant="mrl_2stage",
                transform=False,
                name="__mirror_check__",
            )
            # production 과 동일한 dense/sparse 벡터를 그대로 재사용(재임베딩 금지)
            mirror_res = self._raw_search(mirror_cfg, [(dense, sparse)], 5)
            mirror_ids = [
                str(p.payload.get("chunk_id")) for p in mirror_res.points
            ]

            report.append(
                {
                    "query_id": qid,
                    "prod_ids": prod_ids,
                    "mirror_ids": mirror_ids,
                    # 충실도 판정 기준: top-5 집합 일치
                    "set_match": set(prod_ids) == set(mirror_ids),
                    # 참고용: 정확한 순서 일치(동점 비결정성으로 흔들릴 수 있음)
                    "order_match": prod_ids == mirror_ids,
                }
            )
        return report


def default_ablation_matrix(collection: str) -> list[RetrievalConfig]:
    """최소 ablation 매트릭스.

    {dense_mrl_2stage, sparse, hybrid} × {rerank off, on}
      + dense_long_only × {rerank off}   (MRL/256-truncation 검증용 대조)
    """
    configs: list[RetrievalConfig] = []
    for mode in ("dense", "sparse", "hybrid"):
        for rerank in (False, True):
            configs.append(
                RetrievalConfig(mode=mode, rerank=rerank, collection=collection)
            )
    configs.append(
        RetrievalConfig(
            mode="dense",
            rerank=False,
            collection=collection,
            dense_variant="long_only",
        )
    )
    return configs
