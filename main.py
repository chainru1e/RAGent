# ════════════════════════════════════════════════════════════════════════════
# main.py - 메인 실행 파일
# ════════════════════════════════════════════════════════════════════════════

from core.rag_system import QdrantRAGSystem
from core.document_loader import DocumentLoader
from pathlib import Path


def main():
    '''메인 파이프라인'''
    
    print("\n" + "=" * 70)
    print("AI Agent 장기 기억 시스템 - 메인 실행")
    print("=" * 70)
    
    # 1️⃣ RAG 시스템 초기화
    rag = QdrantRAGSystem()
    
    # 2️⃣ 문서 로드
    print("\n1️⃣ 문서 로드")
    print("-" * 70)
    
    loader = DocumentLoader()
    documents_dir = "./data/documents"
    
    # 디렉토리 생성 (없으면)
    Path(documents_dir).mkdir(parents=True, exist_ok=True)
    
    documents = loader.load_documents(documents_dir)
    
    if not documents:
        print("\n⚠️  문서가 없습니다!")
        print(f"'{documents_dir}' 디렉토리에 .txt 또는 .json 파일을 추가하세요.")
        return
    
    # 3️⃣ 문서 수집 (벡터화 및 저장)
    print("\n2️⃣ 문서 수집 및 벡터화")
    print("-" * 70)
    
    total_chunks = rag.ingest_documents(documents)
    
    # 4️⃣ 통계
    print("\n3️⃣ 저장소 통계")
    print("-" * 70)
    
    stats = rag.get_stats()
    print(f"컬렉션: {stats['collection']}")
    print(f"총 청크: {stats['total_points']}")
    print(f"벡터 차원: {stats['vector_size']}")
    
    # 5️⃣ 검색 테스트
    print("\n4️⃣ 검색 테스트")
    print("-" * 70)
    
    test_queries = [
        "벡터 데이터베이스",
        "빠른 검색",
        "임베딩"
    ]
    
    for query in test_queries:
        print(f"\n🔍 쿼리: '{query}'")
        
        results = rag.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n  [{i}] {result['chunk_id']} (점수: {result['score']:.4f})")
            print(f"       {result['text'][:60]}...")
    
    # 6️⃣ 문맥 통합 (LLM 입력용)
    print("\n5️⃣ 문맥 통합")
    print("-" * 70)
    
    final_results = rag.search("벡터 검색", top_k=3)
    context = rag.summarize_results(final_results)
    
    print("\n📝 LLM 입력용 문맥:")
    print(context)
    
    print("\n" + "=" * 70)
    print("✅ 실행 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()