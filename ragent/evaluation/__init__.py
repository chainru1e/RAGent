"""오프라인 검색(retrieval) 평가 하네스.

RAGent 의 핵심 검색 파이프라인(Hybrid 검색 → reranker)의 순위 품질을 ranx 로
측정한다. 평가 대상은 Reranker.rerank 직후의 순위 리스트이며(현재 코드에는
MetadataExpander 가 없으므로 이것이 곧 "확장 이전" 순위 리스트), rerank on/off 를
ablation 축으로 두어 "순수 hybrid 순위(off)"와 "재정렬 순위(on)"를 모두 본다.

이 패키지는 기존 retrieval/embedding/vectordb 코드를 import·재사용만 하며 절대
수정하지 않는다. Qdrant 는 read-only query_points 로만 접근한다(upsert/delete 없음).
"""
