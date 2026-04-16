"""
config.py
─────────
모든 하이퍼파라미터와 경로 설정을 한 곳에서 관리합니다.
"""

import torch

# ── Device ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Architecture ──────────────────────────────────
INPUT_DIM = 768        # 원본 임베딩 차원 (e.g., text-embedding-3-small)
HIDDEN_DIM = 512       # 첫 번째 은닉층
PROJECTION_DIM = 128   # 최종 투영 차원

DROPOUT_RATE = 0.1

# ── Training ────────────────────────────────────────────
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 200
BATCH_SIZE = 64

# ── NCA Loss ────────────────────────────────────────────
NCA_TEMPERATURE = 1.0

# ── Scheduler ───────────────────────────────────────────
T_MAX = EPOCHS          # CosineAnnealingLR 주기

# ── Early Stopping ──────────────────────────────────────
PATIENCE = 15

# ── Paths ───────────────────────────────────────────────
MODEL_SAVE_PATH = "nca_model.pt"
VECTOR_STORE_PATH = "vector_store.json"

# ── Retriever ───────────────────────────────────────────
INTENT_BOOST = 1.2     # 의도 매칭 문서 점수 부스트 배수
TOP_K = 5              # 검색 결과 상위 K개
