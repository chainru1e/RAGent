import sys
import yaml
import torch
import argparse
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from datetime import datetime

# config.py에서 레지스트리 함수 가져오기
from configs.config import get_adapter_components
from trainer import AdapterTrainer
from data.io import load_from_pt

def main():
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description="Train Embedding Adapters")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml file")
    args = parser.parse_args()

    # 설정 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # YAML 설정에서 어댑터 타입 읽기
    adapter_type = config["adapter"]["type"]

    # config.py의 레지스트리를 이용해 클래스와 손실 함수 동적 로드
    AdapterClass, loss_fn = get_adapter_components(adapter_type)

    # data 폴더 안 pt 파일 전부 로드 후 합치기
    pt_files = list(Path("data").glob("*.pt"))
    if not pt_files:
        print("[오류] data 폴더 내 pt 파일 없음.")
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
    print(f"[Config] adapter type: {adapter_type}")
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

    # 동적으로 불러온 어댑터 & 트레이너 초기화
    adapter = AdapterClass(dim=dim)
    trainer = AdapterTrainer(adapter, loss_fn, config, device)

    trainer.fit(dataloader)

    # 학습된 가중치 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"checkpoints/{adapter_type}_adapter_{timestamp}.pt"
    trainer.save(save_path)
    print(f"[저장] 가중치가 저장되었습니다: {save_path}")

if __name__ == "__main__":
    main()