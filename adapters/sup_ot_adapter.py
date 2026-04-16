import torch
import torch.nn as nn
import torch.nn.functional as F

class SupOTAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        
    def forward(self, x):
        # 입력 벡터를 변환 후 L2 정규화하여 방향만 유지
        adapted = self.projection(x)
        return F.normalize(adapted, p=2, dim=1)

def compute_soft_supervised_cost_matrix(x, y, labels_x, labels_y, discount_factor=0.2):
    """
    타입이 같을 경우 거리 비용을 비율(%)로 줄여주는 비용 행렬
    """
    # 기본 코사인 거리 계산
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    base_cost = 1.0 - torch.mm(x_norm, y_norm.t())
    
    # 타입 마스크 생성 (같으면 1.0, 다르면 0.0)
    same_type_mask = (labels_x.unsqueeze(1) == labels_y.unsqueeze(0)).float()
    
    # 할인 적용 (같은 타입일 경우 비용 감소)
    cost_matrix = base_cost * (1.0 - (discount_factor * same_type_mask))
    
    return torch.clamp(cost_matrix, min=0.0)

def sinkhorn_wasserstein_loss(x, y, labels_x, labels_y, epsilon=0.05, max_iters=50, discount_factor=0.2):
    """
    Sinkhorn 알고리즘을 이용한 의도 기반 Wasserstein Distance 계산
    """
    # 1. Soft Nudging 비용 행렬 계산
    cost_matrix = compute_soft_supervised_cost_matrix(
        x, y, labels_x, labels_y, discount_factor=discount_factor
    )
    
    batch_size = x.size(0)

    # 2. 균등 분포 초기화
    mu = torch.ones(batch_size, requires_grad=False, device=x.device) / batch_size
    nu = torch.ones(batch_size, requires_grad=False, device=y.device) / batch_size

    # 3. 커널 행렬 K
    K = torch.exp(-cost_matrix / epsilon)
    u = torch.ones_like(mu)

    # 4. Sinkhorn Iteration
    for _ in range(max_iters):
        v = nu / torch.matmul(K.t(), u)
        u = mu / torch.matmul(K, v)

    # 5. 최적 이동 계획 및 최종 비용 계산
    pi = u.unsqueeze(1) * K * v.unsqueeze(0)
    wasserstein_dist = torch.sum(pi * cost_matrix)
    
    return wasserstein_dist

def sup_ot_loss(adapted, base_emb, labels, epsilon=0.05, max_iters=20, discount_factor=0.2, alpha=0.8):
    # 1. Target 분포 생성을 위해 배치 내부를 무작위 셔플
    indices = torch.randperm(adapted.size(0), device=adapted.device)
    y = adapted[indices]
    labels_y = labels[indices]
    
    # 2. 의도 기반 최적 운송 Loss
    ot_loss = sinkhorn_wasserstein_loss(
        adapted, y, labels, labels_y, 
        epsilon=epsilon, max_iters=max_iters, discount_factor=discount_factor
    )
    
    # 3. 원본 공간 보존 Loss (L2 정규화 후 MSE)
    norm_base = F.normalize(base_emb, p=2, dim=1)
    reg_loss = F.mse_loss(adapted, norm_base)
    
    # 4. 최종 Loss
    return (alpha * ot_loss) + ((1 - alpha) * reg_loss)