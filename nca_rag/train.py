"""
train.py
────────
NCA 모델 학습 실행 스크립트.
실행: python train.py
"""

import numpy as np
import torch

from config import (
    DEVICE, INPUT_DIM, HIDDEN_DIM, PROJECTION_DIM, DROPOUT_RATE,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, BATCH_SIZE,
    NCA_TEMPERATURE, T_MAX, PATIENCE, MODEL_SAVE_PATH,
)
from nca_rag.NCA_Adapter import NCANetwork, NCALoss
from trainer import NCATrainer


def generate_synthetic_data(num_intents: int = 6,
                            samples_per_intent: int = 80,
                            dim: int = 768) -> tuple:
    """
    학습 검증용 합성 데이터를 생성합니다.
    
    각 의도별로 클러스터 중심(centroid)을 만들고,
    가우시안 노이즈를 추가하여 현실적인 분포를 시뮬레이션합니다.
    """
    np.random.seed(42)
    
    intent_names = [
        "상품_문의", "배송_조회", "환불_요청",
        "기술_지원", "계정_관리", "일반_질문",
    ][:num_intents]

    all_embeddings = []
    all_labels = []

    for i in range(num_intents):
        # 의도별 클러스터 중심 생성
        centroid = np.random.randn(dim).astype(np.float32) * 3.0

        # 중심 주변에 노이즈 샘플 분포
        noise = np.random.randn(samples_per_intent, dim).astype(np.float32) * 0.5
        samples = centroid + noise

        all_embeddings.append(samples)
        all_labels.extend([i] * samples_per_intent)

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels, dtype=np.int64)

    # 셔플
    indices = np.random.permutation(len(labels))
    embeddings = embeddings[indices]
    labels = labels[indices]

    print(f"  📊 Synthetic Data Generated:")
    print(f"     Intents: {intent_names}")
    print(f"     Samples: {len(labels)} ({samples_per_intent}/intent)")
    print(f"     Dim: {dim}D")
    print()

    return (
        torch.tensor(embeddings, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        intent_names,
    )


def evaluate_clustering(model, embeddings, labels,
                        intent_names, device):
    """학습된 모델의 클러스터 분리도를 평가합니다."""
    model.eval()
    with torch.no_grad():
        projected = model(embeddings.to(device)).cpu().numpy()

    labels_np = labels.numpy()
    num_intents = len(intent_names)

    # 의도별 중심 계산
    centroids = []
    for i in range(num_intents):
        mask = labels_np == i
        centroids.append(projected[mask].mean(axis=0))
    centroids = np.array(centroids)

    # 의도 내 평균 거리 (Intra-cluster)
    intra_dists = []
    for i in range(num_intents):
        mask = labels_np == i
        cluster_points = projected[mask]
        dists = np.linalg.norm(cluster_points - centroids[i], axis=1)
        intra_dists.append(dists.mean())

    # 의도 간 평균 거리 (Inter-cluster)
    inter_dists = []
    for i in range(num_intents):
        for j in range(i + 1, num_intents):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_dists.append(dist)

    avg_intra = np.mean(intra_dists)
    avg_inter = np.mean(inter_dists)
    separation_ratio = avg_inter / (avg_intra + 1e-10)

    print(f"  📈 Cluster Evaluation:")
    print(f"     Avg Intra-cluster Distance : {avg_intra:.4f}")
    print(f"     Avg Inter-cluster Distance : {avg_inter:.4f}")
    print(f"     Separation Ratio (↑ better): {separation_ratio:.2f}")
    print()

    for i, name in enumerate(intent_names):
        print(f"     [{name:>8s}] intra-dist: {intra_dists[i]:.4f}")
    print()


def main():
    print()
    print(f"  🔧 Device: {DEVICE}")
    print(f"  🏗️  Architecture: {INPUT_DIM}D → {HIDDEN_DIM}D "
          f"→ {HIDDEN_DIM//2}D → {PROJECTION_DIM}D")
    print()

    # ── 1. 데이터 준비 ──
    embeddings, labels, intent_names = generate_synthetic_data(
        num_intents=6,
        samples_per_intent=80,
        dim=INPUT_DIM,
    )

    # ── 2. 모델 & 손실 함수 생성 ──
    model = NCANetwork(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        projection_dim=PROJECTION_DIM,
        dropout=DROPOUT_RATE,
    )
    criterion = NCALoss(temperature=NCA_TEMPERATURE)

    # ── 3. Trainer 초기화 ──
    trainer = NCATrainer(
        model=model,
        criterion=criterion,
        device=DEVICE,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        t_max=T_MAX,
        patience=PATIENCE,
    )

    # ── 4. 학습 실행 ──
    history = trainer.fit(
        embeddings=embeddings,
        labels=labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True,
    )

    # ── 5. 모델 저장 ──
    trainer.save(MODEL_SAVE_PATH)

    # ── 6. 클러스터 분리도 평가 ──
    evaluate_clustering(model, embeddings, labels, intent_names, DEVICE)

    print(f"  🎉 Done! Stopped at epoch {history['stopped_epoch']}")
    print(f"     Best loss: {history['best_loss']:.6f}")
    print(f"     Model: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
