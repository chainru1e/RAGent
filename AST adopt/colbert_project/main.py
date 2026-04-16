# file: colbert_project/main.py
import sys
import os
from colbert_rag import ColBERTRAGSystem

def main():
    """메인 실행 함수"""
    
    # 1️⃣ RAG 시스템 초기화
    print("🚀 ColBERT RAG 시스템 초기화...")
    rag = ColBERTRAGSystem()
    
    # 2️⃣ 문서 로드 (data/documents 폴더에서)
    documents = []
    doc_dir = os.path.join(os.path.dirname(__file__), "data", "documents")
    
    if os.path.exists(doc_dir):
        for filename in os.listdir(doc_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(doc_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    documents.append(f.read())
        print(f"✅ {len(documents)}개 문서 로드 완료")
    else:
        # 샘플 문서 사용
        documents = [
            "Qdrant는 벡터 데이터베이스로, 대규모 데이터에서 빠른 검색을 제공한다.",
            "LoRA(Low-Rank Adaptation)는 효율적인 파인튜닝 방법이다.",
            "ColBERT는 토큰 레벨 임베딩으로 정교한 검색을 한다.",
            "Hybrid 검색은 Dense + Sparse 결합 방식이다.",
        ]
        print("⚠️  문서 폴더 없음. 샘플 데이터 사용.")
    
    # 3️⃣ 인덱싱
    print("\n📥 문서 인덱싱 중...")
    rag.index_documents(documents)
    
    # 4️⃣ 인터랙티브 검색
    print("\n🔍 검색 시작 (종료: 'quit' 입력)")
    while True:
        query = input("\n쿼리 입력: ").strip()
        if query.lower() == "quit":
            print("👋 프로그램 종료")
            break
        
        results = rag.retrieve(query, top_k=3, use_hybrid=True)
        
        print(f"\n📊 상위 {len(results)}개 결과:")
        for r in results:
            print(f"{r['rank']}. [Hybrid: {r['hybrid_score']:.4f}]")
            print(f"   {r['text']}\n")
    
    # 5️⃣ 저장
    print("\n💾 인덱스 저장 중...")
    rag.save_index("./colbert_storage/colbert.pkl")
    print("✅ 저장 완료")


if __name__ == "__main__":
    main()
