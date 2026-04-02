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
        alpha: float = ALPHA,
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
            dense_model_name='all-MiniLM-L6-v2',
            alpha=alpha
        )
        
        self.storage = QdrantStorage(
            collection_name=collection_name,
            vector_size=384,
            path=qdrant_path
        )
        
        print(f"\n✅ RAG 시스템 초기화 완료\n")
    
    def ingest_documents(self, documents: List[Dict]) -> int:
        '''
        문서 수집 및 벡터화 → Qdrant 저장
        '''
        
        print("\n" + "=" * 70)
        print("📥 문서 수집 및 벡터화")
        print("=" * 70)
        
        total_chunks = 0
        
        for doc in documents:
            doc_id = doc.get("doc_id")
            text = doc.get("text", "")
            doc_metadata = doc.get("metadata", {})
            
            print(f"\n📄 {doc_id} ({len(text)} 글자)")
            
            # 청킹
            chunks = self.chunker.chunk_text(text, doc_id)
            print(f"   청킹: {len(chunks)}개 청크")
            
            # 임베딩 + 저장
            chunks_to_save = []
            
            for i, chunk in enumerate(chunks):
                dense_vector = self.embedding.get_dense_embedding(chunk["text"])
                
                chunks_to_save.append({
                    "chunk_id": chunk["chunk_id"],
                    "vector": dense_vector,
                    "text": chunk["text"],
                    "metadata": {
                        "original_doc_id": chunk["original_doc_id"],
                        "chunk_num": chunk["chunk_num"],
                        **doc_metadata
                    }
                })
            
            # Qdrant에 저장
            self.storage.add_points_batch(chunks_to_save)
            total_chunks += len(chunks)
        
        print(f"\n" + "=" * 70)
        print(f"✅ 수집 완료: 총 {total_chunks}개 청크 저장")
        print("=" * 70)
        
        return total_chunks
    
    def search(
        self,
        query: str,
        top_k: int = SEARCH_TOP_K,
        filter_dict: Dict = None
    ) -> List[Dict]:
        '''
        쿼리 검색 (Qdrant)
        '''
        
        print(f"\n🔍 검색: '{query}'")
        print("-" * 70)
        
        query_vector = self.embedding.get_dense_embedding(query)
        
        results = self.storage.search(
            query_vector=query_vector,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        formatted_results = []
        
        for chunk_id, score, text in results:
            if score >= SEARCH_SCORE_THRESHOLD:
                formatted_results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": score,
                    "metadata": self._extract_metadata(chunk_id)
                })
        
        print(f"결과: {len(formatted_results)}개")
        
        return formatted_results
    
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
