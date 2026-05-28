"""턴 단위 인덱싱 오케스트레이션 + 신뢰성 책임.

chunker / intent_classifier / embedder / vectordb 호출을 한 함수로 묶고,
실패 격리 + transient 재시도 + 영구 실패 로깅을 캡슐화한다. 신뢰성 로직을
어댑터별 훅 레이어가 아닌 인덱싱 함수 자체에 캡슐화한다. 어떤 어댑터의
호출 경로든 동일한 신뢰성 보장을 균등하게 받는다.
"""

import json
import logging
import hashlib
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

from ragent.config import FAILED_CHUNKS_FILE, ensure_dirs
from ragent.models.parsed_message import NormalizedMessage
from ragent.modules.chunking_modules import Chunker
from ragent.modules.contextual_retrieval_modules import (
    ContextualEnricher,
    serialize_turn,
)
from ragent.modules.embedding_modules import HybridEmbedding
from ragent.modules.intent_classifying_modules import IntentClassifier
from ragent.vectordb_client import QdrantStorage

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
    intent_classifier: IntentClassifier,
    embedder: HybridEmbedding,
    vectordb: QdrantStorage,
    contextual_enricher: ContextualEnricher | None = None,
    workspace_id: str | None = None,
) -> int:
    """단일 턴을 chunk → classify → (enrich) → embed → upsert 하나의 단위로 인덱싱.

    실패 처리 정책:
    - 어떤 단계든 예외가 올라오면 silent fail 차단을 위해 잡아 로그/기록 후 0 반환
    - 외부 자원 호출(embed_batch, qdrant upsert) 은 transient 실패 대비 retry
    - 영구 실패는 failed_chunks.jsonl 에 append 하여 추후 진단 가능

    contextual_enricher 가 주어지면 코드 청크 각각에 대해 turn 전체를 문서로
    한 맥락 문장을 생성해 chunk.metadata.context_prefix 에 보관한다. 임베딩
    입력만 "prefix + payload" 로 합본하고 chunk.payload 와 vectordb 저장값은
    원본을 유지한다(검색 결과로 prefix 가 노출되지 않도록).
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

        for chunk in chunks:
            chunk.metadata.workspace_id = workspace_id
            chunk.metadata.source_kind = chunk.metadata.source_kind or "conversation"

        context_chunk = chunks[0]
        intent = intent_classifier.classify(context_chunk.payload).category

        if contextual_enricher is not None and len(chunks) > 1:
            turn_text = serialize_turn(turn)
            t0 = time.time()
            enriched = 0
            for code_chunk in chunks[1:]:
                prefix = contextual_enricher.generate_prefix(
                    turn_text, code_chunk.payload
                )
                if prefix:
                    code_chunk.metadata.context_prefix = prefix
                    enriched += 1
            logger.info(
                "indexer: contextual prefixes generated for %d/%d code chunks in %.2fs (%s)",
                enriched, len(chunks) - 1, time.time() - t0, turn_summary,
            )

        texts = [
            f"{c.metadata.context_prefix}\n\n{c.payload}"
            if c.metadata.context_prefix
            else c.payload
            for c in chunks
        ]
        vectors = _retry(
            lambda: embedder.embed_batch(texts),
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


def index_file_snapshot(
    *,
    absolute_path: str,
    relative_path: str,
    workspace_id: str,
    session_id: str,
    chunker: Chunker,
    embedder: HybridEmbedding,
    vectordb: QdrantStorage,
) -> int:
    """변경된 파일의 현재 디스크 상태를 버전형 file_snapshot 으로 인덱싱한다.

    파일이 삭제된 경우에는 기존 current snapshot 만 비활성화한다. 파일이 존재할
    때는 동일 content_hash 중복 인덱싱을 생략하고, 기존 current snapshot 을
    is_current=false 로 내린 뒤 새 snapshot 을 is_current=true 로 저장한다.
    """
    path = Path(absolute_path)
    op_summary = f"session={session_id}, file={relative_path}"

    try:
        if not path.exists():
            return _retry(
                lambda: vectordb.deactivate_current_snapshots(
                    workspace_id=workspace_id,
                    file_path=relative_path,
                ),
                op_name=f"qdrant deactivate deleted snapshot ({op_summary})",
            )

        if not path.is_file():
            logger.info("snapshot indexer: skipping non-file path (%s)", op_summary)
            return 0

        content = path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        previous_hash = _retry(
            lambda: vectordb.current_snapshot_hash(
                workspace_id=workspace_id,
                file_path=relative_path,
            ),
            op_name=f"qdrant current hash ({op_summary})",
        )
        if previous_hash == content_hash:
            logger.info("snapshot indexer: unchanged file skipped (%s)", op_summary)
            return 0

        snapshot_version = _retry(
            lambda: vectordb.next_snapshot_version(
                workspace_id=workspace_id,
                file_path=relative_path,
            ),
            op_name=f"qdrant next snapshot version ({op_summary})",
        )
        snapshot_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{workspace_id}:{relative_path}:{snapshot_version}:{content_hash}",
            )
        )
        indexed_at = datetime.now(timezone.utc).isoformat()

        chunks = chunker.process_file_snapshot(
            workspace_id=workspace_id,
            file_path=relative_path,
            content=content,
            content_hash=content_hash,
            snapshot_id=snapshot_id,
            snapshot_version=snapshot_version,
            indexed_at=indexed_at,
        )
        if not chunks:
            logger.warning("snapshot indexer: no chunks produced (%s)", op_summary)
            return 0

        texts = [c.payload for c in chunks]
        vectors = _retry(
            lambda: embedder.embed_batch(texts),
            op_name=f"embed_batch snapshot ({op_summary})",
        )
        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector

        _retry(
            lambda: vectordb.deactivate_current_snapshots(
                workspace_id=workspace_id,
                file_path=relative_path,
            ),
            op_name=f"qdrant deactivate current snapshot ({op_summary})",
        )
        count = _retry(
            lambda: vectordb.add_points_batch(chunks),
            op_name=f"qdrant upsert snapshot ({op_summary})",
        )
        logger.info("snapshot indexer: upserted %d chunks (%s)", count, op_summary)
        return count
    except UnicodeDecodeError as exc:
        logger.warning("snapshot indexer: skipping non-text file (%s): %s", op_summary, exc)
        return 0
    except Exception as exc:
        logger.exception("snapshot indexer: permanent failure (%s)", op_summary)
        _record_failure(session_id, f"snapshot:{relative_path}", exc)
        return 0
