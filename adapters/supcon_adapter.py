# https://arxiv.org/pdf/2004.11362

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.net(x)
    
def supcon_infonce_loss(embeddings, labels, temperature=0.07):
    """
    Supervised Contrastive InfoNCE Loss
    
    Args:
        embeddings: (N, D) - 어댑터 출력 벡터 (정규화 전)
        labels:     (N,)   - 각 샘플의 클래스 레이블
        temperature: float - 온도 파라미터
    
    Returns:
        scalar loss
    """
    device = embeddings.device
    N = embeddings.size(0)

    # L2 정규화 (코사인 유사도 기반 비교)
    z = F.normalize(embeddings, dim=1)          # (N, D)

    # 유사도 행렬
    sim_matrix = torch.matmul(z, z.T) / temperature   # (N, N)

    # 자기 자신과의 유사도는 -inf 마스킹 (분모·분자 모두에서 제외)
    self_mask = torch.eye(N, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(self_mask, float('-inf'))

    # 같은 레이블 쌍 마스크 (자기 자신 제외)
    labels = labels.unsqueeze(1)                       # (N, 1)
    pos_mask = (labels == labels.T) & ~self_mask       # (N, N)

    # 양성 쌍이 없는 샘플은 손실 계산에서 제외
    valid = pos_mask.any(dim=1)                        # (N,)
    if not valid.any():
        return embeddings.sum() * 0.0                  # gradient 유지용 zero loss

    sim_matrix = sim_matrix[valid]                     # (M, N)
    pos_mask   = pos_mask[valid]                       # (M, N)

    # log-sum-exp (분모): 자기 자신 제외한 전체
    log_denom = torch.logsumexp(sim_matrix, dim=1)     # (M,)

    # 각 양성 쌍에 대한 log P 평균
    # pos_mask 위치의 sim 값만 골라 평균
    loss_per_sample = -(
        sim_matrix.masked_fill(~pos_mask, 0).sum(dim=1)   # 양성 sim 합
        - pos_mask.sum(dim=1) * log_denom                  # 양성 개수 × log_denom
    ) / pos_mask.sum(dim=1)                                # 양성 개수로 나눠 평균

    return loss_per_sample.mean()