"""
models.py
─────────
NCANetwork : 768D → 128D 비선형 투영 MLP
NCALoss    : Neighborhood Components Analysis 손실 함수
"""

import torch
import torch.nn as nn


class NCAAdapter(nn.Module):
    """
    Deep Non-Linear MLP for NCA Projection.
    
    Architecture (Bottleneck):
        Input(768) → Linear(512) → BN → ReLU → Dropout
                   → Linear(256) → BN → ReLU → Dropout
                   → Linear(128) → L2-Normalize
    """

    def __init__(self, input_dim: int = 768,
                 hidden_dim: int = 512,
                 projection_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1: input_dim → hidden_dim
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 2: hidden_dim → hidden_dim // 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 3: hidden_dim // 2 → projection_dim
            nn.Linear(hidden_dim // 2, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.network(x)
        # L2 정규화로 단위 구 위에 투영
        return nn.functional.normalize(projected, p=2, dim=-1)


class NCALoss(nn.Module):
    """
    Neighborhood Components Analysis Loss.

    같은 의도(intent)의 임베딩은 가까이,
    다른 의도의 임베딩은 멀리 밀어내는 metric learning 손실.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (N, D) 투영된 임베딩
            labels:     (N,)   정수 레이블 (intent ID)
        Returns:
            scalar loss
        """
        # 쌍별 제곱 거리: ||x_i - x_j||^2
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (N,N,D)
        sq_distances = (diff ** 2).sum(dim=-1)                    # (N,N)

        # 소프트맥스 기반 유사도 (자기 자신 제외)
        logits = -sq_distances / self.temperature                 # (N,N)
        mask_self = torch.eye(len(embeddings),
                              device=embeddings.device).bool()
        logits.masked_fill_(mask_self, float('-inf'))

        log_probs = torch.log_softmax(logits, dim=-1)            # (N,N)

        # 같은 라벨 마스크 (자기 자신 제외)
        label_match = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N,N)
        label_match.masked_fill_(mask_self, False)

        # 같은 라벨 쌍의 log-prob 합산 → 최대화 (= -loss 최소화)
        # 같은 라벨이 없는 샘플은 제외
        has_positive = label_match.sum(dim=-1) > 0
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device,
                                requires_grad=True)

        nca_score = (log_probs * label_match.float()).sum(dim=-1)  # (N,)
        loss = -nca_score[has_positive].mean()

        return loss
