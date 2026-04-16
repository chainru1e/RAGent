# file: colbert_project/colbert_encoder.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import math

class ColBERTEncoder(nn.Module):
    """
    ColBERT: Contextualized Late Interaction over BERT
    - 토큰별 임베딩 생성 (Dense Passage Retrieval 보다 정교)
    - 쿼리와 문서 간 "늦은 상호작용" 구현
    """
    
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=128,  # ColBERT 표준: 128D
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # 기본 BERT 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name).to(device)
        
        # 원본 임베딩 차원 (all-MiniLM-L6-v2는 384D)
        hidden_dim = self.base_model.config.hidden_size
        
        # 저차원 투영 층 (384D → 128D)
        self.linear_projection = nn.Linear(hidden_dim, embedding_dim)
        self.linear_projection.to(device)
        
    def forward(self, texts, return_tokens=False):
        """
        Args:
            texts: List[str], 문서 또는 쿼리 리스트
            return_tokens: bool, 토큰 정보 반환 여부
        
        Returns:
            embeddings: (batch_size, seq_len, embedding_dim)
            token_ids: Optional, 토크나이제이션 결과
        """
        # 토크나이제이션 (패딩 없음 → 토큰 길이가 다양함)
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=False,  # 동적 패딩은 나중에 배치 단계에서 처리
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        # BERT 인코더를 통과 (마지막 은닉층 추출)
        with torch.no_grad():
            outputs = self.base_model(**encoded)
            last_hidden = outputs.last_hidden_state  # (seq_len, 384)
        
        # 저차원 투영 (384D → 128D)
        embeddings = self.linear_projection(last_hidden)  # (seq_len, 128)
        
        # L2 정규화 (코사인 유사도를 내적으로 계산하기 위함)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        if return_tokens:
            return embeddings, encoded
        return embeddings
    
    def encode_document(self, doc_text):
        """
        문서 인코딩 (토큰별 벡터 생성)
        
        Returns:
            doc_vec: (seq_len, 128) 텐서
            tokens: 토큰 문자열 리스트
        """
        encoded = self.tokenizer(
            doc_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(**encoded)
            last_hidden = outputs.last_hidden_state
        
        embeddings = self.linear_projection(last_hidden)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # 토큰 추출
        token_ids = encoded['input_ids'][0].cpu().tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        return embeddings[0], tokens  # (seq_len, 128), List[str]
    
    def encode_query(self, query_text):
        """
        쿼리 인코딩
        
        Returns:
            query_vec: (seq_len, 128) 텐서
            tokens: 토큰 문자열 리스트
        """
        return self.encode_document(query_text)


# ----- 테스트 코드 -----
if __name__ == "__main__":
    encoder = ColBERTEncoder()
    
    doc = "Qdrant는 벡터 데이터베이스로, 대규모 데이터에서 빠른 검색을 제공한다."
    query = "벡터 데이터베이스"
    
    doc_vec, doc_tokens = encoder.encode_document(doc)
    query_vec, query_tokens = encoder.encode_query(query)
    
    print(f"문서 토큰: {doc_tokens}")
    print(f"문서 벡터 형태: {doc_vec.shape}")  # (seq_len, 128)
    print(f"\n쿼리 토큰: {query_tokens}")
    print(f"쿼리 벡터 형태: {query_vec.shape}")  # (seq_len, 128)
