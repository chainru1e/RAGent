"""검색 평가 오케스트레이터 — GT 로드 → 진단 → ablation 실행 → ranx 리포트.

실행 예::

    python -m ragent.evaluation.evaluate --collection -Users-choseungbeom-Downloads-RAGent
    python -m ragent.evaluation.evaluate --collection <NAME> --k 5 --depth 20
    python -m ragent.evaluation.evaluate --collection <NAME> --transform   # production 충실 모드

설계 메모
=========
- 평가 대상 = Reranker.rerank 직후 순위(현재 코드에 MetadataExpander 없음 →
  retrieve() 출력이 곧 "확장 이전" 순위). rerank on/off 가 ablation 축.
- @k=5 는 고정 주입 컷오프 M 이 아니라 후보 풀/주입 상한 limit=5 에서 따온 값.
  실제 주입은 static_cutoff(0.3)+dynamic_cutoff(0.1) 로 가변 → cutoff 진단으로 별도 보고.
- depth=N=20 으로 검색해 R@20(천장)도 함께 본다. R@20 − R@5 = 더 나은 순위로
  회복 가능한 recall.

이 스크립트는 오프라인 도구다. 구조적 오류(GT/컬렉션/Qdrant 누락)는 loud fail 로
즉시 멈추고 원인을 출력한다. 개별 쿼리 retrieval 실패만 스킵·카운트한다.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.metadata
import json
import logging
import sys
from pathlib import Path

from ragent.evaluation.retrieval_runner import (
    DYNAMIC_CUTOFF_DROP,
    DYNAMIC_CUTOFF_MIN_CHUNKS,
    STATIC_CUTOFF_THRESHOLD,
    EvalRetriever,
    RetrievalConfig,
    RunResult,
    default_ablation_matrix,
)
from ragent.modules.retrieval_modules import dynamic_cutoff, static_cutoff

logger = logging.getLogger("ragent.evaluation")

DEFAULT_GT = Path(__file__).with_name("ground_truth.jsonl")


def _die(message: str, code: int = 2) -> "NoReturn":  # type: ignore[name-defined]
    """loud fail: 표준에러에 원인을 찍고 0 이 아닌 코드로 종료한다."""
    print(f"\n[FATAL] {message}\n", file=sys.stderr)
    raise SystemExit(code)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def load_ground_truth(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """ground_truth.jsonl 을 로드한다.

    Returns:
        queries:  {query_id: query_text}        (라벨 유무와 무관, 전체)
        labels:   {query_id: {relevant_chunk_id}}  (relevant 가 비지 않은 것만)
    """
    if not path.exists():
        _die(f"ground truth file not found: {path}")

    queries: dict[str, str] = {}
    labels: dict[str, set[str]] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            _die(f"invalid JSON at {path}:{lineno}: {exc}")

        qid = obj.get("query_id")
        query = obj.get("query")
        rel = obj.get("relevant_chunk_ids", [])
        if not qid or not isinstance(query, str) or not query:
            _die(
                f"malformed record at {path}:{lineno} — "
                f"need non-empty 'query_id' and 'query' (got {obj!r})"
            )
        if qid in queries:
            _die(f"duplicate query_id {qid!r} at {path}:{lineno}")
        if not isinstance(rel, list):
            _die(f"'relevant_chunk_ids' must be a list at {path}:{lineno}")

        queries[qid] = query
        rel_ids = {str(c) for c in rel}
        if rel_ids:
            labels[qid] = rel_ids
    return queries, labels


# --------------------------------------------------------------------------
# 진단(diagnostics)
# --------------------------------------------------------------------------
def diagnose_labels(labels: dict[str, set[str]], k: int) -> dict:
    """쿼리당 relevant 개수 분포 vs k. k 초과 시 R@k 상한 캡 경고."""
    counts = {qid: len(ids) for qid, ids in labels.items()}
    over_k = {qid: n for qid, n in counts.items() if n > k}
    print("\n## 진단 1) relevant 개수 분포 vs k")
    if not counts:
        print("  (라벨된 쿼리 없음)")
    else:
        dist = sorted(counts.values())
        print(f"  라벨된 쿼리: {len(counts)}개, relevant 개수 min/median/max = "
              f"{dist[0]}/{dist[len(dist)//2]}/{dist[-1]}")
        if over_k:
            print(f"  ⚠️  relevant > k(={k}) 인 쿼리 {len(over_k)}개 → 이 쿼리들은 "
                  f"Recall@{k} 상한이 캡됩니다(top-{k} 에 다 못 담음):")
            for qid, n in over_k.items():
                print(f"      - {qid}: relevant={n} > k={k}")
        else:
            print(f"  모든 쿼리의 relevant ≤ k(={k}) — Recall@{k} 캡 없음.")
    return {"counts": counts, "over_k": over_k}


def diagnose_existence(
    retriever: EvalRetriever, collection: str, labels: dict[str, set[str]]
) -> dict:
    """GT 의 chunk_id 가 실제 컬렉션에 존재하는지 검증(read-only)."""
    all_ids = sorted({cid for ids in labels.values() for cid in ids})
    print("\n## 진단 2) GT chunk_id 의 컬렉션 실재 검증")
    if not all_ids:
        print("  (검증할 chunk_id 없음 — 라벨 미입력)")
        return {"missing": []}
    found = retriever.chunk_ids_exist(collection, all_ids)
    missing = [cid for cid in all_ids if cid not in found]
    if missing:
        print(f"  ⚠️  컬렉션 {collection!r} 에 없는 chunk_id {len(missing)}/"
              f"{len(all_ids)}개:")
        for cid in missing:
            owners = [q for q, ids in labels.items() if cid in ids]
            print(f"      - {cid}  (queries: {', '.join(owners)})")
    else:
        print(f"  모든 GT chunk_id({len(all_ids)}개)가 컬렉션에 존재합니다.")
    return {"missing": missing}


def diagnose_empty(results: list[RunResult]) -> dict:
    """검색 결과가 빈 쿼리 / 스킵된 쿼리 보고."""
    print("\n## 진단 3) 빈 결과 / 스킵된 쿼리")
    out = {}
    for rr in results:
        if rr.empty or rr.skipped:
            print(f"  [{rr.name}] empty={rr.empty or '-'}  skipped={rr.skipped or '-'}")
        out[rr.name] = {"empty": rr.empty, "skipped": rr.skipped}
    if not any(rr.empty or rr.skipped for rr in results):
        print("  모든 config 에서 빈 결과·스킵 없음.")
    return out


def diagnose_cutoff(
    prod_like: RunResult, labels: dict[str, set[str]], k: int
) -> dict:
    """cutoff 진단 — production-like(hybrid+rerank on) 기준.

    @k 지표는 '후보 순위 품질'이지 '실제 주입량'이 아니다. 실제 주입은 reranker
    점수에 대한 static_cutoff(0.3)+dynamic_cutoff(0.1) 로 가변이다. 여기서는
    쿼리당 평균 주입 개수와, cutoff 로 인한 recall 손실(R@k 대비)을 보고한다.
    """
    print("\n## 진단 4) cutoff 진단 (production-like: %s)" % prod_like.name)
    if not labels:
        print("  (라벨 미입력 — recall 손실 계산 생략)")
        return {}

    inj_counts: list[int] = []
    inj_recalls: list[float] = []
    topk_recalls: list[float] = []
    per_query = {}
    for qid, relevant in labels.items():
        scores = prod_like.run.get(qid, {})
        # reranker 점수 내림차순 = 순위. (id, score) 튜플로 cutoff 함수 재사용.
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        after_static = static_cutoff(ordered, STATIC_CUTOFF_THRESHOLD)
        after_dynamic = dynamic_cutoff(
            after_static, DYNAMIC_CUTOFF_DROP, DYNAMIC_CUTOFF_MIN_CHUNKS
        )
        injected = [cid for cid, _ in after_dynamic]
        topk = [cid for cid, _ in ordered[:k]]

        inj_rec = len(set(injected) & relevant) / len(relevant)
        topk_rec = len(set(topk) & relevant) / len(relevant)
        inj_counts.append(len(injected))
        inj_recalls.append(inj_rec)
        topk_recalls.append(topk_rec)
        per_query[qid] = {
            "injected_count": len(injected),
            "injected_recall": inj_rec,
            "recall@k": topk_rec,
        }

    n = len(labels)
    avg_inj = sum(inj_counts) / n
    mean_inj_rec = sum(inj_recalls) / n
    mean_topk_rec = sum(topk_recalls) / n
    loss = mean_topk_rec - mean_inj_rec
    print(f"  쿼리당 평균 주입 개수(cutoff 통과분): {avg_inj:.2f}")
    print(f"  주입분 mean recall            : {mean_inj_rec:.4f}")
    print(f"  R@{k} (top-{k} mean recall)    : {mean_topk_rec:.4f}")
    print(f"  cutoff 로 인한 recall 손실     : {loss:.4f}  (R@{k} − 주입분 recall)")
    return {
        "avg_injected_count": avg_inj,
        "mean_injected_recall": mean_inj_rec,
        f"mean_recall@{k}": mean_topk_rec,
        "recall_loss_from_cutoff": loss,
        "per_query": per_query,
    }


# --------------------------------------------------------------------------
# ranx 평가
# --------------------------------------------------------------------------
def build_metrics(k: int, depth: int) -> list[str]:
    """ranx 0.3.21 검증된 지표 문자열명."""
    metrics = [
        f"precision@{k}",
        f"recall@{k}",
        f"recall@{depth}",
        f"mrr@{k}",
        f"ndcg@{k}",
        f"hit_rate@{k}",
    ]
    # k==depth 등으로 중복되면 정리
    seen: list[str] = []
    for m in metrics:
        if m not in seen:
            seen.append(m)
    return seen


def report_to_markdown(report) -> str:
    """ranx Report.to_dict() → markdown 표."""
    d = report.to_dict()
    metrics = d["metrics"]
    names = d["model_names"]
    header = "| Model | " + " | ".join(metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)
    rows = [header, sep]
    for name in names:
        cells = [name]
        for m in metrics:
            v = d[name]["scores"][m] if "scores" in d[name] else d[name][m]
            cells.append(f"{v:.4f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="RAGent retrieval 평가 (ranx)")
    parser.add_argument("--collection", required=True, help="평가 대상 Qdrant 컬렉션명")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT,
                        help=f"ground_truth.jsonl 경로 (기본 {DEFAULT_GT})")
    parser.add_argument("--k", type=int, default=5, help="평가 @k (기본 5)")
    parser.add_argument("--depth", type=int, default=20, help="검색 depth N (기본 20)")
    parser.add_argument("--transform", action="store_true",
                        help="QueryTransformer(LLM) ON — production 충실 모드(LLM 서버 필요)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).with_name("reports"),
                        help="리포트 출력 디렉토리")
    args = parser.parse_args()

    if args.depth < args.k:
        _die(f"--depth({args.depth}) 는 --k({args.k}) 이상이어야 합니다.")

    ranx_version = importlib.metadata.version("ranx")
    timestamp = _dt.datetime.now().isoformat(timespec="seconds")

    print("=" * 70)
    print("RAGent retrieval 평가 (ranx)")
    print("=" * 70)
    print(f"collection : {args.collection}")
    print(f"@k         : {args.k}   (고정 주입 컷오프 M 이 아님 — limit=5/cutoff 기반)")
    print(f"depth N    : {args.depth}")
    print(f"transform  : {args.transform}  (production 은 항상 ON; 기본 OFF=재현성)")
    print(f"ranx       : {ranx_version}")
    print(f"timestamp  : {timestamp}")
    print("\n참고: 평가 대상은 Reranker.rerank 직후 순위(MetadataExpander 부재). "
          "rerank off=순수 hybrid 순위, on=재정렬 순위.")

    # --- 로드 & 연결 (loud fail) ---
    queries, labels = load_ground_truth(args.gt)
    if not queries:
        _die(f"ground truth 에 쿼리가 없습니다: {args.gt}")

    try:
        retriever = EvalRetriever()
        retriever.ensure_collection(args.collection)
    except Exception as exc:  # noqa: BLE001
        _die(f"Qdrant 연결/컬렉션 확인 실패: {type(exc).__name__}: {exc}")

    # --- 라벨/존재 진단 (라벨 불필요한 검색 진단 전에) ---
    diag_labels = diagnose_labels(labels, args.k)
    diag_exist = diagnose_existence(retriever, args.collection, labels)

    # --- 미러 충실도 검증 ---
    print("\n## 진단 0) hybrid 미러 충실도 (vs production staged_hybrid_search)")
    try:
        mirror = retriever.verify_mirror_fidelity(queries, args.collection, sample=3)
        print(
            "  판정 기준 = top-5 chunk_id 집합 일치(set_match). 순서(order_match)는 "
            "Qdrant 동점 비결정성으로 흔들릴 수 있어 참고용."
        )
        for m in mirror:
            mark = "✅" if m["set_match"] else "❌"
            order = "동일" if m["order_match"] else "동점-순서차"
            print(
                f"  {mark} {m['query_id']}: set_match={m['set_match']} "
                f"(order={order})"
            )
            if not m["set_match"]:
                print(f"      prod  : {m['prod_ids']}")
                print(f"      mirror: {m['mirror_ids']}")
        if mirror and not all(m["set_match"] for m in mirror):
            print("  ⚠️  미러 top-5 집합이 production 과 불일치 — hybrid run 해석에 주의.")
        elif mirror:
            print("  ✅ 미러 충실: 모든 표본에서 top-5 집합 일치(순서차는 동점 비결정성).")
    except Exception as exc:  # noqa: BLE001 - 진단 실패가 평가 전체를 막지 않음
        mirror = [{"error": f"{type(exc).__name__}: {exc}"}]
        print(f"  ⚠️  미러 검증 실패(평가는 계속): {type(exc).__name__}: {exc}")

    # --- ablation 실행 ---
    configs = default_ablation_matrix(args.collection)
    if args.transform:
        for c in configs:
            c.transform = True
            c.name = c.default_name()
    print(f"\n## ablation 실행 — {len(configs)}개 config, depth={args.depth}")
    results: list[RunResult] = []
    total_skipped = 0
    for cfg in configs:
        print(f"  - {cfg.name} ...", flush=True)
        rr = retriever.run_config(cfg, queries, depth=args.depth)
        results.append(rr)
        total_skipped += len(rr.skipped)

    diag_empty = diagnose_empty(results)

    # --- cutoff 진단 (production-like = hybrid + rerank on) ---
    prod_like = next(
        (r for r in results
         if r.config.mode == "hybrid" and r.config.rerank), None
    )
    diag_cut = diagnose_cutoff(prod_like, labels, args.k) if prod_like else {}

    # --- ranx 평가 ---
    print("\n" + "=" * 70)
    if not labels:
        print("⚠️  라벨된 쿼리가 없습니다(relevant_chunk_ids 가 모두 비어 있음).")
        print("    ground_truth.jsonl 에 relevant_chunk_ids 를 채운 뒤 다시 실행하면")
        print("    ranx 지표(precision/recall/mrr/ndcg/hit_rate)가 계산됩니다.")
        print("    (이번 실행은 구조 진단·미러검증·검색 동작 확인까지만 수행)")
        if total_skipped:
            print(f"\n총 {total_skipped}개 쿼리 스킵됨(개별 retrieval 실패).")
        _save_report(args, ranx_version, timestamp, configs, None, None,
                     {"labels": diag_labels, "existence": diag_exist,
                      "empty": diag_empty, "cutoff": diag_cut, "mirror": mirror},
                     total_skipped)
        return

    from ranx import Qrels, Run, compare

    qrels = Qrels({qid: {cid: 1 for cid in ids} for qid, ids in labels.items()})
    runs = []
    for rr in results:
        # 라벨된 쿼리만, 그리고 비어있지 않은 결과만 ranx 에 넘긴다.
        run_dict = {
            qid: rr.run[qid]
            for qid in labels
            if rr.run.get(qid)
        }
        if not run_dict:
            print(f"  (run 제외: {rr.name} — 라벨된 쿼리에 대한 결과 없음)")
            continue
        runs.append(Run(run_dict, name=rr.name))

    if not runs:
        _die("라벨된 쿼리에 대해 비어있지 않은 run 이 하나도 없습니다.")

    metrics = build_metrics(args.k, args.depth)
    report = compare(qrels=qrels, runs=runs, metrics=metrics, max_p=0.05)

    print("\n## ranx 비교 리포트")
    print(report)
    md = report_to_markdown(report)
    print("\n## markdown")
    print(md)

    if total_skipped:
        print(f"\n총 {total_skipped}개 쿼리 스킵됨(개별 retrieval 실패).")

    _save_report(args, ranx_version, timestamp, configs, report, md,
                 {"labels": diag_labels, "existence": diag_exist,
                  "empty": diag_empty, "cutoff": diag_cut, "mirror": mirror},
                 total_skipped)


def _save_report(args, ranx_version, timestamp, configs, report, md, diagnostics,
                 total_skipped) -> None:
    """report.json / report.md / report.tex 저장."""
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "collection": args.collection,
        "k": args.k,
        "depth": args.depth,
        "transform": args.transform,
        "ranx_version": ranx_version,
        "timestamp": timestamp,
        "k_note": "고정 주입 컷오프 M 아님 — limit=5/cutoff(static 0.3 + dynamic 0.1) 기반",
        "eval_target": "Reranker.rerank 직후 순위 (MetadataExpander 부재)",
        "configs": [
            {"name": c.name, "mode": c.mode, "rerank": c.rerank,
             "dense_variant": c.dense_variant, "transform": c.transform}
            for c in configs
        ],
        "total_skipped_queries": total_skipped,
    }
    payload = {
        "meta": meta,
        "diagnostics": diagnostics,
        "ranx_report": report.to_dict() if report is not None else None,
    }
    json_path = out_dir / "report.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n저장: {json_path}")

    if md is not None:
        md_path = out_dir / "report.md"
        md_path.write_text(
            f"# RAGent retrieval 평가\n\n"
            f"- collection: `{args.collection}`\n- @k: {args.k}, depth: {args.depth}\n"
            f"- transform: {args.transform}\n- ranx: {ranx_version}\n"
            f"- timestamp: {timestamp}\n\n{md}\n",
            encoding="utf-8",
        )
        print(f"저장: {md_path}")

    if report is not None:
        try:
            tex_path = out_dir / "report.tex"
            tex_path.write_text(report.to_latex(), encoding="utf-8")
            print(f"저장: {tex_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  to_latex 저장 실패: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
