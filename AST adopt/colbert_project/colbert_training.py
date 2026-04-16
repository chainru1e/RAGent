# file: colbert_project/colbert_training.py
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from colbert_encoder import ColBERTEncoder
from late_interaction import LateInteraction
import torch.nn.functional as F

class ColBERTTrainingDataset(Dataset):
    """
    ColBERT 파인튜닝용 데이터셋
    형식: (query, positive_doc, [negative_docs])
    """
    
    def __init__(self, examples):
        """
        Args:
            examples: List[{
                'query': str,
                'positive': str,
                'negatives': List[str] (선택사항)
            }]
        """
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'query': ex['query'],
            'positive': ex['positive'],
            'negatives': ex.get('negatives', [])
        }


class ColBERTInfoNCELoss(torch.nn.Module):
    """
    ColBERT를 위한 InfoNCE 손실 함수
    """
    
    def __init__(self, encoder: ColBERTEncoder, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
    
    def forward(self, query_vecs, positive_vecs, negative_vecs_batch):
        """
        Args:
            query_vecs: (q_len, 128) 쿼리 토큰 임베딩
            positive_vecs: (p_len, 128) 긍정 문서 임베딩
            negative_vecs_batch: List[(n_len, 128)] 부정 문서 임베딩들
        
        Returns:
            loss: scalar
        """
        # ColBERT 점수 계산
        pos_score = LateInteraction.compute_colbert_score(query_vecs, positive_vecs)
        
        neg_scores = [
            LateInteraction.compute_colbert_score(query_vecs, neg_vecs)
            for neg_vecs in negative_vecs_batch
        ]
        
        # InfoNCE 손실 계산
        pos_exp = torch.exp(torch.tensor(pos_score, device=query_vecs.device) / self.temperature)
        neg_exp_sum = sum([
            torch.exp(torch.tensor(s, device=query_vecs.device) / self.temperature) for s in neg_scores
        ])
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8))
        
        return loss


class ColBERTTrainer:
    """ColBERT 모델 학습"""
    
    def __init__(self, encoder: ColBERTEncoder, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.encoder = encoder
        self.device = device
        self.criterion = ColBERTInfoNCELoss(encoder)
        self.optimizer = AdamW(encoder.linear_projection.parameters(), lr=1e-5)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """한 epoch 학습"""
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            query = batch['query'][0]  # 첫 번째 예시만 사용 (배치 간소화)
            positive = batch['positive'][0]
            negatives = batch['negatives'][0] if batch['negatives'][0] else []
            
            # 인코딩
            query_vec, _ = self.encoder.encode_query(query)
            pos_vec, _ = self.encoder.encode_document(positive)
            neg_vecs = [
                self.encoder.encode_document(neg)[0] for neg in negatives
            ] if negatives else [torch.randn_like(pos_vec)]
            
            # 손실 계산
            loss = self.criterion(query_vec, pos_vec, neg_vecs)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx+1}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch} 완료 - Avg Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def train(self, train_data: list, epochs=5, batch_size=4):
        """전체 학습"""
        dataset = ColBERTTrainingDataset(train_data)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
        
        print("✅ 학습 완료!")


# ----- 테스트 코드 -----
if __name__ == "__main__":
    # 학습 데이터 준비
    train_data = [
        {
            'query': "벡터 데이터베이스",
            'positive': "Qdrant는 벡터 데이터베이스다.",
            'negatives': ["파이썬 프로그래밍", "머신러닝 기초"]
        },
        {
            'query': "LoRA 파인튜닝",
            'positive': "LoRA는 모델 파라미터를 효율적으로 조정한다.",
            'negatives': ["전체 모델 학습", "고정 벡터 임베딩"]
        },
    ]
    
    # 학습 실행
    encoder = ColBERTEncoder()
    trainer = ColBERTTrainer(encoder)
    trainer.train(train_data, epochs=2)
