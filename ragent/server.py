"""Local RAG pipeline server.

This server keeps heavyweight RAG objects alive between Claude Code hook calls.
Hook handlers should call this process over HTTP instead of initializing models
inside each hook invocation.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import RLock
from typing import Any

from ragent import config
from ragent.logging_config import setup_logging
from ragent.models.chunk import Chunk
from ragent.modules.chunking_modules import Chunker
from ragent.modules.contextual_retrieval_modules import ContextualEnricher
from ragent.modules.embedding_modules import HybridEmbedding
from ragent.modules.indexing_modules import index_turn
from ragent.modules.intent_classifying_modules import IntentClassifier
from ragent.modules.retrieval_modules import Retriever
from ragent.vectordb_client import QdrantStorage

logger = logging.getLogger("ragent.server")

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8765


class RAGentServer:
    def __init__(self):
        self.chunker = Chunker()
        self.intent_classifier = IntentClassifier()
        self.embedder = HybridEmbedding()
        self.contextual_enricher = ContextualEnricher()
        self._vectordb_cache: dict[str, QdrantStorage] = {}
        self._lock = RLock()
        self._save_queue: queue.Queue[dict[str, Any]] = queue.Queue()

        # /stats 엔드포인트가 노출할 작업 상태. launcher 콘솔 출력 외에는 사용처 없음.
        self._stats_lock = threading.Lock()
        self._current_task: dict[str, Any] | None = None
        self._last_completed: dict[str, Any] | None = None
        self._search_lock = threading.Lock()
        self._active_searches: list[dict[str, Any]] = []
        self._last_search: dict[str, Any] | None = None

        self._save_worker_thread = threading.Thread(
            target=self._save_worker,
            name="ragent-save-worker",
            daemon=True,
        )
        self._save_worker_thread.start()

    def enqueue_save(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate the request and put it on the save queue. Returns immediately.

        The single save worker drains the queue serially so the in-process
        embedding model (not thread-safe) and the local LLM (HTTP-serialized
        anyway) are never hit concurrently.
        """
        session_id = request.get("session_id", "")
        transcript_path = request.get("transcript_path", "")

        if not session_id:
            return {"ok": False, "queued": False, "reason": "missing_session_id"}

        if not transcript_path:
            return {"ok": False, "queued": False, "reason": "missing_transcript_path"}

        self._save_queue.put(request)
        return {"ok": True, "queued": True, "pending": self._save_queue.qsize()}

    def _save_worker(self) -> None:
        while True:
            request = self._save_queue.get()
            session_id = request.get("session_id", "?")
            started_at = time.time()
            with self._stats_lock:
                self._current_task = {
                    "kind": "save",
                    "session_id": session_id,
                    "started_at": started_at,
                }
            status = "ok"
            chunks_indexed = 0
            try:
                chunks_indexed = self._process_save(request)
            except Exception:
                status = "failed"
                logger.exception("Save worker: unhandled error for %s", request)
            finally:
                duration = time.time() - started_at
                with self._stats_lock:
                    self._last_completed = {
                        "kind": "save",
                        "session_id": session_id,
                        "duration_s": round(duration, 3),
                        "finished_at": time.time(),
                        "status": status,
                        "chunks_indexed": chunks_indexed,
                    }
                    self._current_task = None
                self._save_queue.task_done()

    def _process_save(self, request: dict[str, Any]) -> int:
        session_id = request.get("session_id", "")
        transcript_path = request.get("transcript_path", "")

        with self._lock:
            parser = self._build_parser(transcript_path)
            last_turn = parser.parse_last_turn()

            if not last_turn:
                logger.warning("Save: no turns found in transcript %s", transcript_path)
                return 0

            # 파일 단위 Contextual Retrieval 경로 ②: 세션 전체에서 latest Write
            # 본문을 모아 index_turn 에 넘긴다. enrichment 가 꺼져 있으면 불필요.
            file_snapshots = (
                parser.build_file_snapshots()
                if config.ENABLE_CONTEXTUAL_RETRIEVAL
                else None
            )

            vectordb = self._get_vectordb(transcript_path)
            count = index_turn(
                turn=last_turn,
                session_id=session_id,
                chunker=self.chunker,
                intent_classifier=self.intent_classifier,
                embedder=self.embedder,
                vectordb=vectordb,
                contextual_enricher=(
                    self.contextual_enricher
                    if config.ENABLE_CONTEXTUAL_RETRIEVAL
                    else None
                ),
                file_snapshots=file_snapshots,
            )

        logger.info("Save: indexed %d chunks for session %s", count, session_id)
        return count

    def search(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id", "")
        transcript_path = request.get("transcript_path", "")
        prompt = request.get("prompt", "")

        if not session_id:
            return {"ok": False, "chunks": [], "context": "", "reason": "missing_session_id"}

        if not transcript_path:
            return {"ok": False, "chunks": [], "context": "", "reason": "missing_transcript_path"}

        if not prompt:
            return {"ok": False, "chunks": [], "context": "", "reason": "missing_prompt"}

        t0 = time.time()
        entry = {"session_id": session_id, "started_at": t0}
        with self._search_lock:
            self._active_searches.append(entry)

        status = "ok"
        chunks: list[Chunk] = []
        context = ""
        try:
            with self._lock:
                vectordb = self._get_vectordb(transcript_path)
                retriever = Retriever(
                    vectordb=vectordb,
                    embedder=self.embedder,
                )
                chunks = retriever.retrieve(prompt)
            context = self._format_context_for_claude(chunks)
        except Exception:
            status = "failed"
            raise
        finally:
            duration = time.time() - t0
            with self._search_lock:
                try:
                    self._active_searches.remove(entry)
                except ValueError:
                    pass
                self._last_search = {
                    "session_id": session_id,
                    "result_count": len(chunks),
                    "context_len": len(context),
                    "duration_s": round(duration, 3),
                    "finished_at": time.time(),
                    "status": status,
                }

        logger.info("Search: retrieved %d chunks for session %s", len(chunks), session_id)
        return {
            "ok": True,
            "chunks": [self._chunk_to_dict(chunk) for chunk in chunks],
            "context": context,
        }

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            current = dict(self._current_task) if self._current_task else None
            last_save = dict(self._last_completed) if self._last_completed else None
        if current is not None:
            current["elapsed_s"] = round(time.time() - current["started_at"], 3)

        with self._search_lock:
            active = list(self._active_searches)
            last_search = dict(self._last_search) if self._last_search else None

        currently_searching = None
        if active:
            oldest = min(active, key=lambda e: e["started_at"])
            currently_searching = {
                "session_id": oldest["session_id"],
                "started_at": oldest["started_at"],
                "elapsed_s": round(time.time() - oldest["started_at"], 3),
            }

        return {
            "queue_pending": self._save_queue.qsize(),
            "currently_processing": current,
            "currently_searching": currently_searching,
            "last_completed": last_save,
            "last_search": last_search,
        }

    def start_server(self):
        host = getattr(config, "RAGENT_SERVER_HOST", DEFAULT_SERVER_HOST)
        port = int(getattr(config, "RAGENT_SERVER_PORT", DEFAULT_SERVER_PORT))
        app = self

        class RequestHandler(BaseHTTPRequestHandler):
            # launcher가 주기적으로 폴링하는 경로 — 로그 도배 방지용
            _SILENT_PATHS = ("/stats", "/health")

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._write_json(200, {"ok": True, "status": "healthy"})
                    return

                if self.path == "/stats":
                    self._write_json(200, app.stats())
                    return

                self._write_json(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                try:
                    body = self._read_json()

                    if self.path == "/save":
                        response = app.enqueue_save(body)
                        self._write_json(202, response)
                        return

                    if self.path == "/search":
                        response = app.search(body)
                        self._write_json(200, response)
                        return

                    self._write_json(404, {"ok": False, "error": "not_found"})
                except json.JSONDecodeError:
                    self._write_json(400, {"ok": False, "error": "invalid_json"})
                except Exception as exc:
                    logger.exception("Unhandled server error")
                    self._write_json(500, {"ok": False, "error": str(exc)})

            def log_message(self, format: str, *args: Any) -> None:
                # launcher의 /stats /health 폴링은 DEBUG 라인도 ragent.log에 남지 않도록 침묵.
                request_line = args[0] if args else ""
                if isinstance(request_line, str) and any(
                    f" {p} " in request_line for p in self._SILENT_PATHS
                ):
                    return
                logger.debug(format, *args)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if not raw:
                    return {}
                data = json.loads(raw.decode("utf-8"))
                if not isinstance(data, dict):
                    raise json.JSONDecodeError("expected object", raw.decode("utf-8"), 0)
                return data

            def _write_json(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        httpd = ThreadingHTTPServer((host, port), RequestHandler)
        logger.info("Starting RAGent server on http://%s:%s", host, port)
        httpd.serve_forever()

    def _collection_name_from_transcript(self, transcript_path: str) -> str:
        collection_name = os.path.basename(os.path.dirname(transcript_path))
        return collection_name or "default"

    def _build_parser(self, transcript_path: str):
        from ragent.adapters import get_adapter

        adapter_cls = type(get_adapter(None))
        if adapter_cls.parser_class is None:
            raise RuntimeError(f"{adapter_cls.__name__} has no parser_class")
        return adapter_cls.parser_class(transcript_path)

    def _get_vectordb(self, transcript_path: str) -> QdrantStorage:
        collection_name = self._collection_name_from_transcript(transcript_path)

        if collection_name not in self._vectordb_cache:
            self._vectordb_cache[collection_name] = QdrantStorage(collection_name)

        return self._vectordb_cache[collection_name]

    def _format_context_for_claude(self, chunks: list[Chunk]) -> str:
        if not chunks:
            return ""

        context = "<context>\n"
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.file_path if chunk.metadata.file_path else f"snippet_{i + 1}"
            context += f'<document index="{i + 1}" source="{source}">\n'
            context += f"{chunk.payload}\n"
            context += "</document>\n"

        context += "</context>"
        return context

    def _chunk_to_dict(self, chunk: Chunk) -> dict[str, Any]:
        chunk_type = chunk.metadata.type
        if isinstance(chunk_type, Enum):
            chunk_type = chunk_type.value

        return {
            "chunk_id": chunk.metadata.chunk_id,
            "parent_id": chunk.metadata.parent_id,
            "file_path": chunk.metadata.file_path,
            "type": chunk_type,
            "payload": chunk.payload,
        }


def main() -> None:
    setup_logging()
    server = RAGentServer()
    server.start_server()


if __name__ == "__main__":
    main()
