"""평가 실행 오케스트레이터.

각 쿼리에 대해 3가지 모드로 검색하고, LLM Judge로 점수를 매겨 결과를 반환한다.
"""

from __future__ import annotations

from eval.judge import LLMJudge
from eval.modes import MODES


def run_evaluation(
    queries: list[str],
    vectordb,
    embedder,
    judge: LLMJudge,
    k: int = 5,
) -> dict:
    """
    Returns:
        {
            "Dense Only":        [[q1_scores], [q2_scores], ...],
            "Hybrid":            [[q1_scores], [q2_scores], ...],
            "Hybrid + Reranker": [[q1_scores], [q2_scores], ...],
        }
        각 scores는 상위 k개 청크의 관련도 점수 리스트 (1~5)
    """
    results: dict[str, list[list[int]]] = {mode: [] for mode in MODES}

    total = len(queries) * len(MODES)
    done = 0

    for query in queries:
        print(f"\n[쿼리] {query}")

        for mode_name, retrieve_fn in MODES.items():
            chunks = retrieve_fn(vectordb, embedder, query, k=k)
            texts = [c.payload or "" for c in chunks]

            # 청크가 k개 미만이면 빈 문자열로 채움
            while len(texts) < k:
                texts.append("")

            scores = judge.score_batch(query, texts)
            results[mode_name].append(scores)

            avg = sum(scores) / len(scores) if scores else 0
            done += 1
            print(f"  {mode_name:20s} 평균 {avg:.2f}점  {scores}  ({done}/{total})")

    return results


def compute_metrics(results: dict, k_values: list[int] = [1, 3, 5]) -> dict:
    """
    결과로부터 각 K값의 지표를 계산한다.

    Returns:
        {
            "Dense Only": {
                "avg_score@1": 3.2,
                "avg_score@3": 2.8,
                "avg_score@5": 2.5,
                "relevant_rate@1": 0.5,   # score >= 4 비율
                "relevant_rate@3": 0.4,
                "relevant_rate@5": 0.36,
            },
            ...
        }
    """
    metrics: dict[str, dict] = {}

    for mode, query_scores in results.items():
        metrics[mode] = {}

        for k in k_values:
            all_scores_at_k = [scores[:k] for scores in query_scores]

            # 평균 점수
            flat = [s for scores in all_scores_at_k for s in scores]
            avg = sum(flat) / len(flat) if flat else 0
            metrics[mode][f"avg_score@{k}"] = round(avg, 3)

            # Relevant Rate (score >= 4)
            relevant = sum(1 for s in flat if s >= 4)
            rate = relevant / len(flat) if flat else 0
            metrics[mode][f"relevant_rate@{k}"] = round(rate, 3)

    return metrics
