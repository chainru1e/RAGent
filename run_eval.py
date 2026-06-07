"""RAGent LLM-as-Judge 평가 실행 스크립트.

사용법:
    python run_eval.py --collection <컬렉션명> --api-key <GROQ_API_KEY>
    python run_eval.py  # eval_queries.json 에서 설정 읽기

사전 조건:
    - ragent launcher 실행 중 (Qdrant + 서버들)
    - eval_queries.json 에 collection 이름과 쿼리 입력
    - GROQ_API_KEY 환경변수 설정 or --api-key 인자
    - Groq API 키 발급: https://console.groq.com
"""

import argparse
import json
import os
import sys

# RAGent 패키지 경로 추가 (RAGent 폴더 안으로 옮긴 경우엔 이 줄 불필요)
RAGENT_PATH = r"C:\RAGent"
if RAGENT_PATH not in sys.path:
    sys.path.insert(0, RAGENT_PATH)

from ragent.vectordb_client import QdrantStorage
from ragent.modules.embedding_modules import HybridEmbedding

from eval.judge import LLMJudge
from eval.runner import run_evaluation, compute_metrics
from eval.plot import plot_avg_score, plot_relevant_rate, save_metrics_json


def main():
    parser = argparse.ArgumentParser(description="RAGent LLM-as-Judge 평가")
    parser.add_argument("--collection", type=str, help="Qdrant 컬렉션 이름")
    parser.add_argument("--api-key", type=str, help="Groq API 키")
    parser.add_argument("--queries-file", type=str, default="eval_queries.json")
    parser.add_argument("--k", type=int, default=5, help="Top-K (기본값: 5)")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    args = parser.parse_args()

    # ── 설정 로드 ──────────────────────────────────────────
    with open(args.queries_file, encoding="utf-8") as f:
        config = json.load(f)

    collection = args.collection or config.get("collection")
    if not collection or collection.startswith("여기에"):
        print("ERROR: eval_queries.json 의 collection 값을 실제 Qdrant 컬렉션명으로 바꿔주세요.")
        sys.exit(1)

    queries: list[str] = config["queries"]
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")

    print(f"컬렉션: {collection}")
    print(f"쿼리 수: {len(queries)}")
    print(f"Top-K: {args.k}")
    print()

    # ── 모델 초기화 ────────────────────────────────────────
    print("임베딩 모델 로딩 중...")
    embedder = HybridEmbedding()
    print("Qdrant 연결 중...")
    vectordb = QdrantStorage(collection)
    print("LLM Judge 초기화 중...")
    judge = LLMJudge(api_key=api_key)
    print()

    # ── 평가 실행 ──────────────────────────────────────────
    print("=" * 50)
    print("평가 시작")
    print("=" * 50)
    results = run_evaluation(queries, vectordb, embedder, judge, k=args.k)

    # ── 지표 계산 ──────────────────────────────────────────
    k_values = [1, 3, min(args.k, 5)]
    k_values = sorted(set(k_values))
    metrics = compute_metrics(results, k_values=k_values)

    print("\n" + "=" * 50)
    print("최종 결과")
    print("=" * 50)
    for mode, m in metrics.items():
        print(f"\n{mode}")
        for key, val in m.items():
            print(f"  {key}: {val}")

    # ── 저장 & 시각화 ──────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    save_metrics_json(metrics, os.path.join(args.output_dir, "metrics.json"))

    plot_avg_score(
        metrics, k_values,
        save_path=os.path.join(args.output_dir, "avg_score.png")
    )
    plot_relevant_rate(
        metrics, k_values,
        save_path=os.path.join(args.output_dir, "relevant_rate.png")
    )


if __name__ == "__main__":
    main()
