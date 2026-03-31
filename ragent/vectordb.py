"""ChromaDB wrapper for RAGent."""

import logging

import chromadb

from ragent.config import CHROMA_DIR, COLLECTION_QA, COLLECTION_SUMMARIES, ensure_dirs

logger = logging.getLogger("ragent")


class RAGentDB:
    """ChromaDB wrapper managing qa_pairs and session_summaries collections."""

    def __init__(self) -> None:
        ensure_dirs()
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._qa = self._client.get_or_create_collection(name=COLLECTION_QA)
        self._summaries = self._client.get_or_create_collection(name=COLLECTION_SUMMARIES)

    def index_qa_pair(self, turn: dict) -> None:
        """Index a single Q&A pair. Uses user_uuid as ID for upsert."""
        doc_id = turn.get("user_uuid") or f"turn-{hash(turn['user_prompt'])}"
        document = f"Q: {turn['user_prompt']}\nA: {turn['assistant_response'][:1000]}"
        metadata = {
            "timestamp": turn.get("timestamp", ""),
            "prompt_length": len(turn["user_prompt"]),
            "response_length": len(turn["assistant_response"]),
        }
        self._qa.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata],
        )

    def index_session_summary(
        self, session_id: str, summary: str, metadata: dict | None = None
    ) -> None:
        """Index a session summary."""
        meta = metadata or {}
        meta.setdefault("session_id", session_id)
        self._summaries.upsert(
            ids=[session_id],
            documents=[summary],
            metadatas=[meta],
        )

    def search(
        self,
        query: str,
        collection: str = COLLECTION_QA,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Search a collection by query text."""
        coll = self._qa if collection == COLLECTION_QA else self._summaries
        kwargs: dict = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = coll.query(**kwargs)
        out = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                entry = {"document": doc}
                if results.get("metadatas"):
                    entry["metadata"] = results["metadatas"][0][i]
                if results.get("distances"):
                    entry["distance"] = results["distances"][0][i]
                out.append(entry)
        return out
