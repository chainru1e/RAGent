import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from datetime import datetime

from adapters.supcon_adapter import SupConAdapter, supcon_infonce_loss
from trainer import AdapterTrainer
from data.io import load_from_pt


def main():
    # 설정 로드
    with open("configs/supcon.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data 폴더 안 pt 파일 전부 로드 후 합치기
    pt_files = list(Path("data").glob("*.pt"))
    if not pt_files:
        print("[오류] data 폴더에 pt 파일이 없어요. export.py를 먼저 실행해주세요.")
        sys.exit(1)
    print(f"[로드] {len(pt_files)}개 파일 발견: {[f.name for f in pt_files]}")

    all_embs, all_labels = [], []
    for pt_file in pt_files:
        embs, labels = load_from_pt(pt_file)
        all_embs.append(embs)
        all_labels.append(labels)

    embs   = torch.cat(all_embs,   dim=0)
    labels = torch.cat(all_labels, dim=0)

    # dim 자동 추출
    dim = embs.shape[1]
    print(f"[Config] adapter dim: {dim} (auto-detected from dataset)")
    print(f"[Config] 총 {len(embs)}개 샘플")

    # 카테고리별 샘플 수 및 가중치 계산
    class_counts = torch.bincount(labels)
    print(f"[데이터] 카테고리별 샘플 수: {class_counts.tolist()}")

    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    # 데이터셋 & 데이터로더 초기화
    dataset    = TensorDataset(embs, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        sampler=sampler,
    )

    # 어댑터 & 트레이너 초기화
    adapter = SupConAdapter(dim=dim)
    trainer = AdapterTrainer(adapter, supcon_infonce_loss, config, device)

    trainer.fit(dataloader)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save(f"checkpoints/supcon_adapter_{timestamp}.pt")


if __name__ == "__main__":
    main()