"""Local RAG pipeline server.

This server keeps heavyweight RAG objects alive between Claude Code hook calls.
Hook handlers should call this process over HTTP instead of initializing models
inside each hook invocation.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import RLock
from typing import Any

from ragent import config
from ragent.config import GEMINI_API_KEY, LOG_FILE, ensure_dirs
from ragent.models.chunk import Chunk
from ragent.modules.chunking_modules import Chunker
from ragent.modules.embedding_modules import HybridEmbedding
from ragent.modules.intent_classifying_modules import HybridClassifier
from ragent.modules.retrieval_modules import Reranker, Retriever
from ragent.vectordb import QdrantStorage

logger = logging.getLogger("ragent.server")

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8765


class RAGentServer:
    def __init__(self):
        self.chunker = Chunker()
        self.intent_classifier = HybridClassifier(GEMINI_API_KEY)
        self.embedder = HybridEmbedding()
        self.reranker = Reranker()
        self._vectordb_cache: dict[str, QdrantStorage] = {}
        self._lock = RLock()

    def save(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id", "")
        transcript_path = request.get("transcript_path", "")

        if not session_id:
            return {"ok": False, "indexed": 0, "reason": "missing_session_id"}

        if not transcript_path:
            return {"ok": False, "indexed": 0, "reason": "missing_transcript_path"}

        with self._lock:
            parser = self._build_parser(transcript_path)
            last_turn = parser.parse_last_turn()

            if not last_turn:
                logger.warning("Save: no turns found in transcript %s", transcript_path)
                return {"ok": True, "indexed": 0, "reason": "no_turn_found"}

            chunks = self.chunker.process_turn(last_turn)
            context_chunk = next((chunk for chunk in chunks if chunk.metadata.chunk_id), None)

            if not context_chunk:
                logger.warning("Save: no context chunk found for session %s", session_id)
                return {"ok": True, "indexed": 0, "reason": "no_context_chunk"}

            intent = self.intent_classifier.classify(context_chunk.payload).category
            texts = [chunk.payload for chunk in chunks]
            vectors = self.embedder.embed_batch(texts, batch_size=32)

            for chunk, vector in zip(chunks, vectors):
                chunk.metadata.type = intent
                chunk.vector = vector

            vectordb = self._get_vectordb(transcript_path)
            count = vectordb.add_points_batch(chunks)

        logger.info("Save: indexed %d chunks for session %s", count, session_id)
        return {"ok": True, "indexed": count}

    def save_all(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id", "")
        transcript_path = request.get("transcript_path", "")

        if not session_id:
            return {"ok": False, "indexed": 0, "reason": "missing_session_id"}

        if not transcript_path:
            return {"ok": False, "indexed": 0, "reason": "missing_transcript_path"}

        with self._lock:
            parser = self._build_parser(transcript_path)
            turns = parser.parse_full_transcript()

            if not turns:
                logger.warning("SaveAll: no turns found in transcript %s", transcript_path)
                return {"ok": True, "indexed": 0, "reason": "no_turn_found"}

            all_chunks: list[Chunk] = []
            for turn_idx, turn in enumerate(turns):
                chunks = self.chunker.process_turn(turn)
                if not chunks:
                    continue

                parent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:turn:{turn_idx}"))
                chunks[0].metadata.chunk_id = parent_id
                for code_idx, chunk in enumerate(chunks[1:]):
                    chunk.metadata.parent_id = parent_id
                    chunk.metadata.chunk_id = str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"{parent_id}:code:{code_idx}")
                    )

                intent = self.intent_classifier.classify(chunks[0].payload).category
                for chunk in chunks:
                    chunk.metadata.type = intent
                all_chunks.extend(chunks)

            if not all_chunks:
                logger.warning("SaveAll: no chunks produced from %s", transcript_path)
                return {"ok": True, "indexed": 0, "reason": "no_chunks"}

            texts = [chunk.payload for chunk in all_chunks]
            vectors = self.embedder.embed_batch(texts, batch_size=32)

            for chunk, vector in zip(all_chunks, vectors):
                chunk.vector = vector

            vectordb = self._get_vectordb(transcript_path)
            count = vectordb.add_points_batch(all_chunks)

        logger.info(
            "SaveAll: indexed %d chunks across %d turns for session %s",
            count,
            len(turns),
            session_id,
        )
        return {"ok": True, "indexed": count, "turns": len(turns)}

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

        with self._lock:
            vectordb = self._get_vectordb(transcript_path)
            retriever = Retriever(
                vectordb=vectordb,
                embedder=self.embedder,
                reranker=self.reranker,
            )
            chunks = retriever.retrieve(prompt)

        logger.info("Search: retrieved %d chunks for session %s", len(chunks), session_id)
        return {
            "ok": True,
            "chunks": [self._chunk_to_dict(chunk) for chunk in chunks],
            "context": self._format_context_for_claude(chunks),
        }

    def start_server(self):
        host = getattr(config, "RAGENT_SERVER_HOST", DEFAULT_SERVER_HOST)
        port = int(getattr(config, "RAGENT_SERVER_PORT", DEFAULT_SERVER_PORT))
        app = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health":
                    self._write_json(200, {"ok": True, "status": "healthy"})
                    return

                self._write_json(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                try:
                    body = self._read_json()

                    if self.path == "/save":
                        response = app.save(body)
                        self._write_json(200, response)
                        return

                    if self.path == "/search":
                        response = app.search(body)
                        self._write_json(200, response)
                        return

                    if self.path == "/save_all":
                        response = app.save_all(body)
                        self._write_json(200, response)
                        return

                    self._write_json(404, {"ok": False, "error": "not_found"})
                except json.JSONDecodeError:
                    self._write_json(400, {"ok": False, "error": "invalid_json"})
                except Exception as exc:
                    logger.exception("Unhandled server error")
                    self._write_json(500, {"ok": False, "error": str(exc)})

            def log_message(self, format: str, *args: Any) -> None:
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


def _setup_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    _setup_logging()
    server = RAGentServer()
    server.start_server()


if __name__ == "__main__":
    main()
