"""턴 단위 인덱싱 오케스트레이션 + 신뢰성 책임.

chunker / intent_classifier / embedder / vectordb 호출을 한 함수로 묶고,
실패 격리 + transient 재시도 + 영구 실패 로깅을 캡슐화한다. 어댑터/핸들러
레이어는 이 함수를 호출만 한다 — 신뢰성 로직을 핸들러에 두지 않는다.

신뢰성 책임이 인덱서로 캡슐화되므로, 어댑터별 SessionEnd 같은 "안전망 훅"이
구조적으로 필요 없다. 모든 어댑터(Claude Code, Codex, ...) 가 동일한
index_turn 호출로 신뢰성 보장을 받는다.
"""

import json
import logging
import time
import traceback
from typing import Callable, TypeVar

from ragent.config import FAILED_CHUNKS_FILE, ensure_dirs
from ragent.models.parsed_message import NormalizedMessage
from ragent.modules.chunking_modules import Chunker
from ragent.modules.embedding_modules import HybridEmbedding
from ragent.modules.intent_classifying_modules import HybridClassifier
from ragent.vectordb import QdrantStorage

logger = logging.getLogger("ragent")

T = TypeVar("T")


def _retry(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 0.5, op_name: str) -> T:
    """transient 실패 대비. 2^i * base_delay 백오프로 attempts 회 시도.

    임베딩 API / Qdrant 호출처럼 네트워크/외부 자원 의존 호출에만 적용한다.
    deterministic 한 in-process 연산에는 적용하지 않는다 (재시도해도 같은 실패).
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if i + 1 < attempts:
                delay = base_delay * (2 ** i)
                logger.warning(
                    "indexer: %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    op_name, i + 1, attempts, delay, exc,
                )
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _record_failure(session_id: str, turn_summary: str, exc: Exception) -> None:
    """영구 실패 항목을 ~/.ragent/failed_chunks.jsonl 에 append.

    추후 진단/재처리 도구가 이 파일을 소비할 수 있도록 라인 단위 JSON 으로
    기록한다. 기록 자체가 실패해도 인덱서 흐름은 깨지면 안 되므로 OSError 는
    로그만 남기고 삼킨다.
    """
    ensure_dirs()
    record = {
        "session_id": session_id,
        "turn_summary": turn_summary,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    try:
        with FAILED_CHUNKS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("indexer: failed to write failure record")


def index_turn(
    turn: list[NormalizedMessage],
    *,
    session_id: str,
    chunker: Chunker,
    intent_classifier: HybridClassifier,
    embedder: HybridEmbedding,
    vectordb: QdrantStorage,
) -> int:
    """단일 턴을 chunk → classify → embed → upsert 하나의 단위로 인덱싱.

    실패 처리 정책:
    - 어떤 단계든 예외가 올라오면 silent fail 차단을 위해 잡아 로그/기록 후 0 반환
    - 외부 자원 호출(embed_batch, qdrant upsert) 은 transient 실패 대비 retry
    - 영구 실패는 failed_chunks.jsonl 에 append 하여 추후 진단 가능
    """
    if not turn:
        return 0

    turn_summary = (
        f"session={session_id}, role0={turn[0].role}, len={len(turn)}, "
        f"ts0={turn[0].timestamp}"
    )

    try:
        # deterministic, no transient failure mode → retry unnecessary
        chunks = chunker.process_turn(turn)
        if not chunks:
            logger.warning("indexer: no chunks produced (%s)", turn_summary)
            return 0

        context_chunk = chunks[0]
        # HybridClassifier has internal keyword fallback → external retry unneeded
        intent = intent_classifier.classify(context_chunk.payload).category
        texts = [c.payload for c in chunks]
        vectors = _retry(
            lambda: embedder.embed_batch(texts, batch_size=32),
            op_name=f"embed_batch ({turn_summary})",
        )
        for c, v in zip(chunks, vectors):
            c.metadata.type = intent
            c.vector = v

        count = _retry(
            lambda: vectordb.add_points_batch(chunks),
            op_name=f"qdrant upsert ({turn_summary})",
        )
        logger.info("indexer: upserted %d chunks (%s)", count, turn_summary)
        return count
    except Exception as exc:
        logger.exception("indexer: permanent failure (%s)", turn_summary)
        _record_failure(session_id, turn_summary, exc)
        return 0
