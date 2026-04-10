# ════════════════════════════════════════════════════════════════════════════
# core/rag_system.py - 통합 RAG 시스템
# ════════════════════════════════════════════════════════════════════════════

from typing import List, Dict
import numpy as np
from core.chunking import DocumentChunker
from core.document_loader import DocumentLoader
from core.embedding import HybridEmbedding
from core.qdrant_storage import QdrantStorage
from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    ALPHA, QDRANT_COLLECTION,
    SEARCH_TOP_K, SEARCH_SCORE_THRESHOLD
)


class QdrantRAGSystem:
    '''
    Qdrant 기반 RAG 시스템 (모든 모듈 통합)
    
    파이프라인:
    1. 문서 로드 (document_loader.py)
    2. 청킹 (chunking.py)
    3. 임베딩 (embedding.py)
    4. 저장 (qdrant_storage.py)
    5. 검색 + 반환
    '''
    
    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        qdrant_path: str = "./qdrant_storage"
    ):
        print("\n" + "=" * 70)
        print("🚀 Qdrant RAG 시스템 초기화")
        print("=" * 70)
        
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="semantic"
        )
        
        self.loader = DocumentLoader()
        
        self.embedding = HybridEmbedding(
            dense_model_name='all-MiniLM-L6-v2'
        )
        
        self.storage = QdrantStorage(
            collection_name=collection_name,
            path=qdrant_path
        )
        
        print(f"\n✅ RAG 시스템 초기화 완료\n")
    
    def ingest_documents(self, documents):
        all_texts = [doc["text"] for doc in documents]

        # 🔥 BM25 학습
        self.embedding.train_bm25(all_texts)

        for doc in documents:
            chunks = self.chunker.chunk_text(doc["text"], doc["doc_id"])

            chunks_to_save = []

            for chunk in chunks:
                dense = self.embedding.get_dense_embedding(chunk["text"])
                sparse = self.embedding.get_sparse_vector(chunk["text"])

                chunks_to_save.append({
                    "chunk_id": chunk["chunk_id"],
                    "dense": dense,
                    "sparse": sparse,
                    "text": chunk["text"]
                })

            self.storage.add_points_batch(chunks_to_save)
    
    def search(self, query, top_k=5):
        # 1️⃣ query → embedding
        dense = self.embedding.get_dense_embedding(query)
        sparse = self.embedding.get_sparse_vector(query)

        # 2️⃣ storage에 검색 맡김 (핵심)
        raw_results = self.storage.search(dense, top_k=20)

        # 3️⃣ 결과 포맷 변환
        results = []
        for text, score in raw_results:
            results.append({
                "text": text,
                "score": score
            })

        return results
    
    def summarize_results(self, results: List[Dict]) -> str:
        '''
        검색 결과를 LLM 입력용 문맥으로 통합
        '''
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            part = f"[{i}] (유사도: {result['score']:.4f})\n{result['text']}"
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        '''시스템 전체 통계'''
        return self.storage.get_stats()
    
    def _extract_metadata(self, chunk_id: str) -> Dict:
        '''청크 ID로부터 메타데이터 추출'''
        parts = chunk_id.split("_")
        return {
            "doc_id": parts[0] if len(parts) > 0 else "",
            "chunk_num": int(parts[-1]) if len(parts) > 0 and parts[-1].isdigit() else 0
        }
