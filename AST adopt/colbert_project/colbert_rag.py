# file: colbert_project/colbert_rag.py
import torch
from colbert_storage import ColBERTStorage
from colbert_training import ColBERTTrainer

class ColBERTRAGSystem:
    """ColBERT 기반 RAG 시스템"""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.storage = ColBERTStorage(device=device)
        self.trainer = ColBERTTrainer(self.storage.encoder, device=device)
    
    def index_documents(self, docs):
        """문서 인덱싱"""
        self.storage.add_documents(docs)
    
    def retrieve(self, query, top_k=5, use_hybrid=True):
        """검색"""
        if use_hybrid:
            return self.storage.retrieve_hybrid(query, top_k=top_k)
        else:
            return self.storage.retrieve_colbert(query, top_k=top_k)
    
    def finetune(self, train_data, epochs=5):
        """모델 파인튜닝"""
        self.trainer.train(train_data, epochs=epochs)
    
    def save_index(self, path):
        """인덱스 저장"""
        self.storage.save(path)
    
    def load_index(self, path):
        """인덱스 로드"""
        self.storage.load(path)


# ----- 실제 사용 예시 -----
if __name__ == "__main__":
    import torch
    
    rag = ColBERTRAGSystem()
    
    # 1️⃣ 문서 인덱싱
    documents = [
        "Qdrant는 벡터 데이터베이스로, 대규모 데이터에서 빠른 검색을 제공한다.",
        "LoRA(Low-Rank Adaptation)는 효율적인 파인튜닝 방법이다.",
        "ColBERT는 토큰 레벨 임베딩으로 정교한 검색을 한다.",
        "Hybrid 검색은 Dense + Sparse 결합 방식이다.",
    ]
    
    print("📥 문서 인덱싱...")
    rag.index_documents(documents)
    
    # 2️⃣ 검색
    print("\n🔍 검색 결과:")
    results = rag.retrieve("벡터 데이터베이스 빠른 검색", top_k=3, use_hybrid=True)
    
    for r in results:
        print(f"{r['rank']}. [Hybrid: {r['hybrid_score']:.4f}] {r['text']}")
    
    # 3️⃣ 파인튜닝 (선택사항)
    print("\n📚 모델 파인튜닝...")
    train_data = [
        {
            'query': "빠른 검색",
            'positive': "Qdrant는 벡터 데이터베이스로, 대규모 데이터에서 빠른 검색을 제공한다.",
            'negatives': ["LoRA는 효율적인 파인튜닝 방법이다."]
        }
    ]
    
    rag.finetune(train_data, epochs=2)
    
    # 4️⃣ 저장/로드
    rag.save_index("./colbert_index")
    rag.load_index("./colbert_index")
