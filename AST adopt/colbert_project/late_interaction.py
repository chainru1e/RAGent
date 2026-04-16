# file: colbert_project/late_interaction.py
import torch
import torch.nn.functional as F

class LateInteraction:
    """
    ColBERT의 핵심: 토큰별 유사도 계산 후 최대값 선택
    """
    
    @staticmethod
    def compute_colbert_score(query_vec, doc_vec):
        """
        단일 쿼리-문서 쌍의 ColBERT 점수 계산
        
        Args:
            query_vec: (q_len, 128) 텐서
            doc_vec: (d_len, 128) 텐서
        
        Returns:
            score: float (0.0 ~ 1.0)
        
        공식:
            ColBERT_score = (1/q_len) * Σ_i max_j cos_sim(q_i, d_j)
        """
        # 코사인 유사도 행렬 계산
        # query_vec: (q_len, 128) × doc_vec^T: (128, d_len) → (q_len, d_len)
        similarity_matrix = torch.matmul(query_vec, doc_vec.t())  # (q_len, d_len)
        
        # 각 쿼리 토큰에 대해 최고 유사도 찾기
        max_sims, _ = torch.max(similarity_matrix, dim=1)  # (q_len,)
        
        # 평균 계산
        score = torch.mean(max_sims).item()
        
        return score
    
    @staticmethod
    def batch_compute_colbert_scores(query_vec, doc_vecs):
        """
        배치 단위 점수 계산
        
        Args:
            query_vec: (q_len, 128) 텐서
            doc_vecs: List[(d_len, 128)] 또는 (batch_size, d_len, 128)
        
        Returns:
            scores: List[float] 또는 (batch_size,) 텐서
        """
        if isinstance(doc_vecs, list):
            return [LateInteraction.compute_colbert_score(query_vec, dv) 
                    for dv in doc_vecs]
        else:
            # 배치 처리 (메모리 효율적)
            batch_size = doc_vecs.shape[0]
            scores = []
            for i in range(batch_size):
                score = LateInteraction.compute_colbert_score(query_vec, doc_vecs[i])
                scores.append(score)
            return torch.tensor(scores)


# ----- 테스트 코드 -----
if __name__ == "__main__":
    # 예시 벡터
    query_vec = torch.randn(5, 128)  # 5개 토큰
    doc_vec = torch.randn(20, 128)   # 20개 토큰
    
    # L2 정규화 (중요!)
    query_vec = F.normalize(query_vec, p=2, dim=-1)
    doc_vec = F.normalize(doc_vec, p=2, dim=-1)
    
    score = LateInteraction.compute_colbert_score(query_vec, doc_vec)
    print(f"ColBERT 점수: {score:.4f}")
