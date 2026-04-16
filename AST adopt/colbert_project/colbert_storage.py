# file: colbert_project/colbert_storage.py
import torch
import pickle
import os
from typing import List, Tuple
from colbert_encoder import ColBERTEncoder
from late_interaction import LateInteraction
from rank_bm25 import BM25Okapi
import numpy as np

class ColBERTStorage:
    """
    ColBERT 기반 하이브리드 벡터 저장소
    - Token-level 임베딩 저장
    - Late Interaction 기반 검색
    - BM25 키워드 검색 병렬 지원
    """
    
    def __init__(self, storage_path="./colbert_storage", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # ColBERT 인코더
        self.encoder = ColBERTEncoder(device=device)
        
        # 저장소
        self.documents = []  # 원본 문서 텍스트
        self.doc_embeddings = []  # 토큰별 임베딩
        self.doc_tokens = []  # 각 문서의 토큰
        
        # BM25 (하이브리드 검색용)
        self.bm25 = None
        self.tokenizer = self.encoder.tokenizer
        
    def add_documents(self, docs: List[str], batch_size=32):
        """
        문서 배치 추가
        
        Args:
            docs: 문서 텍스트 리스트
            batch_size: 배치 크기
        """
        print(f"📥 {len(docs)}개 문서 추가 중...")
        
        for i, doc in enumerate(docs):
            # ColBERT 인코딩
            doc_vec, tokens = self.encoder.encode_document(doc)
            
            self.documents.append(doc)
            self.doc_embeddings.append(doc_vec.detach().cpu())  # GPU → CPU 저장
            self.doc_tokens.append(tokens)
            
            if (i + 1) % batch_size == 0:
                print(f"  ✓ {i + 1}/{len(docs)} 완료")
        
        # BM25 학습
        self._train_bm25()
        print("✅ 모든 문서 추가 완료")
        
    def _train_bm25(self):
        """BM25 모델 학습"""
        tokenized_docs = [
            self.tokenizer.tokenize(doc) for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def retrieve_colbert(self, query: str, top_k=10):
        """
        ColBERT 기반 검색
        
        Returns:
            List[(rank, doc_id, score, text)]
        """
        query_vec, _ = self.encoder.encode_query(query)
        query_vec = query_vec.to(self.device)
        
        scores = []
        for i, doc_vec in enumerate(self.doc_embeddings):
            doc_vec = doc_vec.to(self.device)
            score = LateInteraction.compute_colbert_score(query_vec, doc_vec)
            scores.append((i, score))
        
        # 정렬 (점수 내림차순)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score) in enumerate(scores[:top_k], 1):
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "colbert_score": score,
                "text": self.documents[doc_id][:200]  # 처음 200자
            })
        
        return results
    
    def retrieve_hybrid(self, query: str, top_k=10, alpha=0.7):
        """
        하이브리드 검색 (ColBERT + BM25)
        
        Args:
            alpha: ColBERT 가중치 (0.0~1.0)
        """
        # ColBERT 검색
        colbert_results = self.retrieve_colbert(query, top_k=top_k*2)
        colbert_scores = {
            r["doc_id"]: r["colbert_score"] for r in colbert_results
        }
        
        # BM25 검색
        bm25_scores_raw = self.bm25.get_scores(self.tokenizer.tokenize(query))
        bm25_scores = {
            i: score for i, score in enumerate(bm25_scores_raw)
        }
        
        # 점수 정규화
        max_colbert = max(colbert_scores.values()) if colbert_scores else 1.0
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        
        colbert_norm = {k: v/max_colbert for k, v in colbert_scores.items()}
        bm25_norm = {k: v/max_bm25 for k, v in bm25_scores.items()}
        
        # 결합
        hybrid_scores = {}
        all_docs = set(colbert_norm.keys()) | set(bm25_norm.keys())
        
        for doc_id in all_docs:
            c_score = colbert_norm.get(doc_id, 0.0)
            b_score = bm25_norm.get(doc_id, 0.0)
            hybrid_scores[doc_id] = alpha * c_score + (1 - alpha) * b_score
        
        # 정렬
        sorted_results = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_results[:top_k], 1):
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "hybrid_score": score,
                "colbert_score": colbert_scores.get(doc_id, 0.0),
                "bm25_score": bm25_scores_raw[doc_id],
                "text": self.documents[doc_id][:200]
            })
        
        return results
    
    def save(self, path=None):
        """저장소 저장"""
        path = path or os.path.join(self.storage_path, "colbert.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "doc_embeddings": self.doc_embeddings,
                "doc_tokens": self.doc_tokens,
            }, f)
        print(f"✅ 저장소 저장: {path}")
    
    def load(self, path=None):
        """저장소 로드"""
        path = path or os.path.join(self.storage_path, "colbert.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.doc_embeddings = data["doc_embeddings"]
        self.doc_tokens = data["doc_tokens"]
        
        self._train_bm25()
        print(f"✅ 저장소 로드: {path}")


# ----- 테스트 코드 -----
if __name__ == "__main__":
    storage = ColBERTStorage()
    
    # 샘플 문서 추가
    docs = [
        "Qdrant는 벡터 데이터베이스로, 대규모 데이터에서 빠른 검색을 제공한다.",
        "LoRA(Low-Rank Adaptation)는 대형 언어 모델의 파라미터 효율적 미세조정 방법이다.",
        "ColBERT는 토큰 레벨 임베딩을 사용해 정교한 검색을 수행한다.",
        "Hybrid 검색은 Dense 벡터와 Sparse 키워드 검색을 결합한 방식이다.",
    ]
    
    storage.add_documents(docs)
    
    # ColBERT 검색
    print("\n=== ColBERT 검색 ===")
    query = "벡터 데이터베이스 빠른 검색"
    results = storage.retrieve_colbert(query, top_k=3)
    for r in results:
        print(f"{r['rank']}. [{r['colbert_score']:.4f}] {r['text']}")
    
    # Hybrid 검색
    print("\n=== Hybrid 검색 ===")
    results = storage.retrieve_hybrid(query, top_k=3)
    for r in results:
        print(f"{r['rank']}. [ColBERT: {r['colbert_score']:.4f}, BM25: {r['bm25_score']:.4f}] {r['text']}")
