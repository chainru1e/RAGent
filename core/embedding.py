# ════════════════════════════════════════════════════════════════════════════
# core/embedding.py - 임베딩
# ════════════════════════════════════════════════════════════════════════════

import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import DENSE_MODEL, DENSE_DIMENSION, ALPHA
from typing import Dict


class HybridEmbedding:
    '''
    하이브리드 임베딩 (Dense + Sparse)
    
    특징:
    - Dense: all-MiniLM-L6-v2 (384차원)
    - Sparse: BM25Okapi (키워드 기반)
    - 융합: 가중 평균 (alpha 가중치)
    '''
    
    def __init__(
        self,
        dense_model_name: str = DENSE_MODEL,
        alpha: float = ALPHA
    ):
        print(f"✅ HybridEmbedding 초기화")
        print(f"   Dense 모델: {dense_model_name}")
        print(f"   Dense 차원: {DENSE_DIMENSION}")
        print(f"   Alpha (Dense 가중치): {alpha}")
        
        self.dense_model = SentenceTransformer(dense_model_name)
        self.alpha = alpha
        self.model_name = dense_model_name
    
    def get_dense_embedding(self, text: str) -> np.ndarray:
        '''
        단일 텍스트의 Dense 임베딩
        
        Args:
            text: 입력 텍스트
        
        Returns:
            384차원 벡터 (np.ndarray)
        '''
        embedding = self.dense_model.encode(text, convert_to_numpy=True)
        return np.array(embedding, dtype=np.float32)
    
    def embed_batch(self, texts: list) -> np.ndarray:
        '''
        배치 임베딩 (속도 최적화)
        
        Args:
            texts: 텍스트 리스트
        
        Returns:
            (N, 384) 형태의 배열
        '''
        embeddings = self.dense_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return np.array(embeddings, dtype=np.float32)
    
    def get_embedding_info(self) -> Dict:
        '''임베딩 모델 정보'''
        return {
            "model_name": self.model_name,
            "dimension": DENSE_DIMENSION,
            "alpha": self.alpha,
            "device": str(self.dense_model.device)
        }
