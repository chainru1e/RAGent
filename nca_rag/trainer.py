"""
trainer.py
──────────
NCATrainer: 학습 루프, 스케줄러, Early Stopping을 캡슐화합니다.
"""

import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset


class EarlyStopping:
    """Patience 기반 Early Stopping."""

    def __init__(self, patience: int = 15):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None

    def step(self, loss: float, model: nn.Module) -> bool:
        """
        Returns:
            True  → 학습 중단
            False → 계속 진행
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """최적 가중치를 모델에 복원합니다."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class NCATrainer:
    """
    NCANetwork 학습을 담당하는 Trainer.

    Features:
        - AdamW 옵티마이저
        - CosineAnnealingLR 스케줄러
        - Early Stopping
        - 자동 Device 배치 (CPU / CUDA)
    """

    def __init__(self, model: nn.Module,
                 criterion: nn.Module,
                 device: torch.device,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 t_max: int = 200,
                 patience: int = 15):

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.device = device

        self.optimizer = AdamW(model.parameters(),
                               lr=lr,
                               weight_decay=weight_decay)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max)
        self.early_stopping = EarlyStopping(patience=patience)

    def _make_dataloader(self, embeddings: torch.Tensor,
                         labels: torch.Tensor,
                         batch_size: int,
                         shuffle: bool = True) -> DataLoader:
        dataset = TensorDataset(embeddings, labels)
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          drop_last=False)

    def fit(self, embeddings: torch.Tensor,
            labels: torch.Tensor,
            epochs: int = 200,
            batch_size: int = 64,
            verbose: bool = True) -> dict:
        """
        학습을 실행합니다.

        Args:
            embeddings: (N, input_dim) 원본 임베딩
            labels:     (N,) 정수 레이블
            epochs:     최대 에폭 수
            batch_size: 배치 크기
            verbose:    에폭별 로그 출력 여부

        Returns:
            dict with keys: 'train_losses', 'best_loss', 'stopped_epoch'
        """
        loader = self._make_dataloader(embeddings, labels, batch_size)
        train_losses = []

        if verbose:
            print(f"{'='*55}")
            print(f"  NCA Training Start")
            print(f"  Device: {self.device} | Samples: {len(embeddings)}")
            print(f"  Epochs: {epochs} | Batch: {batch_size}")
            print(f"{'='*55}")

        stopped_epoch = epochs
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_emb, batch_lbl in loader:
                batch_emb = batch_emb.to(self.device)
                batch_lbl = batch_lbl.to(self.device)

                self.optimizer.zero_grad()
                projected = self.model(batch_emb)
                loss = self.criterion(projected, batch_lbl)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            train_losses.append(avg_loss)
            self.scheduler.step()

            if verbose and (epoch % 10 == 0 or epoch == 1):
                lr_now = self.scheduler.get_last_lr()[0]
                print(f"  [Epoch {epoch:>4d}/{epochs}]  "
                      f"Loss: {avg_loss:.6f}  LR: {lr_now:.6f}")

            # Early Stopping 체크
            if self.early_stopping.step(avg_loss, self.model):
                stopped_epoch = epoch
                if verbose:
                    print(f"\n  ⏹ Early Stopped at Epoch {epoch} "
                          f"(patience={self.early_stopping.patience})")
                break

        # 최적 가중치 복원
        self.early_stopping.restore_best(self.model)
        if verbose:
            print(f"  ✅ Best Loss: {self.early_stopping.best_loss:.6f}")
            print(f"{'='*55}\n")

        return {
            'train_losses': train_losses,
            'best_loss': self.early_stopping.best_loss,
            'stopped_epoch': stopped_epoch,
        }

    def save(self, path: str) -> None:
        """모델 가중치를 .pt 파일로 저장합니다."""
        torch.save(self.model.state_dict(), path)
        print(f"  💾 Model saved → {path}")

    def load(self, path: str) -> None:
        """저장된 .pt 파일에서 모델 가중치를 복원합니다."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.model.eval()
        print(f"  📂 Model loaded ← {path}")
