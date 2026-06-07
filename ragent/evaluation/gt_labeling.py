"""GT 라벨링 풀 자동 파이프라인 — 쿼리 생성 → relevance 라벨링 → κ 검증.

목적
====
RAGent 검색 평가용 ground truth 를 **사람 개입 없이(선택적)** 만들기 위한 도구.
세 단계로 나뉜다.

1. gen-queries: 코퍼스(라이브 Qdrant 컬렉션)에서 다양성 있게 seed 청크를 뽑고,
   각 seed 가 다룬 작업을 나중에 그 개발자가 되찾으려 물어볼 법한 **backward-
   reference 쿼리**를 LLM 으로 1개씩 생성한다. → ground_truth.jsonl 에 빈 라벨로 기록.
   (leakage 방어를 위해 라벨링과 **분리**되어 있고, seed 청크를 relevant 로 박지 않는다.)

2. label: 각 쿼리에 대해 ablation 매트릭스의 **모든 config** top-k 를 합쳐 pool 을
   만들고(pooling bias 회피), pool 안의 (query, chunk) 쌍을 LLM judge 로 **독립**
   판정해 relevant_chunk_ids 를 채운다.

3. sample / kappa: judge 라벨의 타당성을 표본 κ(Cohen's kappa) 로 방어한다.
   stratified 표본을 뽑아(sample) 사람이 human_label 을 채운 뒤, κ·raw agreement·
   2x2 혼동표를 계산(kappa)한다.

설계 원칙
=========
- gen·judge 모두 Claude (Anthropic API, ``anthropic`` SDK).
- 인증 키는 **전용 env ``RAGENT_JUDGE_API_KEY``** 에서만 읽어 ``Anthropic(api_key=...)``
  에 명시 전달한다. 전역 ``ANTHROPIC_API_KEY`` 는 읽지도 설정하지도 않는다
  (Claude Code 구독 → API 과금 전환 방지). 키가 없으면 gen/label 은 멈추고
  "RAGENT_JUDGE_API_KEY 설정 후 실행" 을 안내한다.
- Qdrant 는 **read-only**(scroll/retrieve/query_points)만 사용한다. upsert/delete 금지.
- production·retrieval_runner·evaluate 의 검색 로직은 **import·재사용만** 한다(수정 금지).
- 호출은 sync + (query_id, chunk_id) 캐시 + 약한 동시성(ThreadPool max~5) +
  rate-limit/일시오류 backoff 재시도. batch API 경로는 구현하지 않는다(추후 옵션).
- 생성 쿼리가 합성(synthetic)임을 사이드카 provenance 에 명시한다.

실행 예::

    export RAGENT_JUDGE_API_KEY=sk-ant-...
    python -m ragent.evaluation.gt_labeling gen-queries --collection <NAME>
    python -m ragent.evaluation.gt_labeling label        --collection <NAME>
    python -m ragent.evaluation.gt_labeling sample
    # (gt_validation_sample.jsonl 의 human_label 을 사람이 채운 뒤)
    python -m ragent.evaluation.gt_labeling kappa
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ragent.config import QDRANT_HOST, QDRANT_PORT

logger = logging.getLogger("ragent.evaluation.gt_labeling")

# --------------------------------------------------------------------------
# 상수 (Opus 로 스왑 용이하도록 한곳에 모음)
# --------------------------------------------------------------------------
GEN_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"

GEN_N = 40                 # 최종 쿼리 수
GEN_OVERSAMPLE = 1.5       # seed 표본을 GEN_N 의 몇 배로 뽑을지(중복 제거 여유분)
POOL_DEPTH = 20            # label pooling 시 각 config 에서 가져올 top-k
KAPPA_SAMPLE_N = 80        # κ 표본 크기(LLM-pos/neg 각 N/2)
SEED = 1234                # 모든 무작위(샘플링/셔플) 재현용 고정 시드

GEN_MAX_TOKENS = 256
JUDGE_MAX_TOKENS = 256
THREADPOOL_MAX = 5         # 약한 동시성
MAX_RETRIES = 5            # rate-limit/일시오류 backoff 재시도 횟수
BACKOFF_BASE = 2.0         # 지수 backoff 기준 초

# seed 다양성: 한 파일에서 너무 많이 뽑히지 않도록 하는 soft 상한.
SEED_FILE_CAP = 3

JUDGE_API_KEY_ENV = "RAGENT_JUDGE_API_KEY"

DEFAULT_COLLECTION = "-Users-choseungbeom-Downloads-RAGent"
DEFAULT_GT = Path(__file__).with_name("ground_truth.jsonl")
DEFAULT_SAMPLE = Path(__file__).with_name("gt_validation_sample.jsonl")

# --------------------------------------------------------------------------
# 프롬프트 (버전·해시는 provenance 에 기록된다)
# --------------------------------------------------------------------------
GEN_PROMPT_VERSION = "gen-v1"
RUBRIC_PROMPT_VERSION = "rubric-v1"

GEN_SYSTEM = "너는 RAGent 평가용 쿼리 생성기다."

GEN_USER_TEMPLATE = """\
작업: 아래 청크가 다룬 작업을 파악하고, 나중에 그 개발자가 그 작업을 되찾으려
물어볼 법한 자연스러운 한 문장 질문을 만든다.

제약:
- intent-level 질문이어야 한다 (예: "저번에 짠 X 어디 있더라", "X 에 버그 있던데 어디였지").
- 청크 텍스트를 그대로 베끼지 말 것. 고유 식별자·코드 라인을 나열하지 말 것.
- 한국어로, 질문 한 문장만.

출력: 질문 텍스트만 (다른 말 없이).

[청크]
{chunk_text}
"""

RUBRIC_SYSTEM = """\
너는 검색 평가용 relevance 판정자다. 쿼리 1개와 청크 1개가 주어진다.

- RELEVANT (1): 청크가 쿼리가 찾는 답·참조 코드·정보를 실제로 담고 있을 때.
- NOT RELEVANT (0): 같은 주제·파일·용어만 등장하고 답을 담지 않으면 0. 애매하면 0.

출력: 근거 1~2문장을 쓰고, 마지막 줄에 정확히 `LABEL: 0` 또는 `LABEL: 1` 만 출력한다."""

JUDGE_USER_TEMPLATE = """\
[쿼리]
{query}

[청크]
{chunk_text}
"""

LABEL_RE = re.compile(r"LABEL:\s*([01])")


# --------------------------------------------------------------------------
# 작은 유틸
# --------------------------------------------------------------------------
def _die(message: str, code: int = 2) -> "NoReturn":  # type: ignore[name-defined]
    """loud fail: 표준에러에 원인을 찍고 0 이 아닌 코드로 종료한다."""
    print(f"\n[FATAL] {message}\n", file=sys.stderr)
    raise SystemExit(code)


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_query(s: str) -> str:
    """근접 중복 판정용 정규화: 소문자 + 공백/문장부호 제거."""
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[?？.!,…\"'`·、。]+", "", s)
    return s


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        _die(f"file not found: {path}")
    out: list[dict] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            _die(f"invalid JSON at {path}:{lineno}: {exc}")
    return out


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """tmp 에 쓰고 os.replace 로 원자 교체 (부분 기록으로 GT 손상 방지)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _cache_path(gt_path: Path) -> Path:
    """라벨 캐시 경로. GT 옆 .gt_cache/ 아래 둔다(.gitignore 됨)."""
    return gt_path.parent / ".gt_cache" / "label_cache.jsonl"


def _cache_key(query_id: str, chunk_id: str) -> str:
    return f"{query_id}\x1f{chunk_id}"


def _load_cache(path: Path) -> dict[str, dict]:
    """라벨 캐시 로드. {cache_key: {query_id, chunk_id, label, rationale, query, text}}."""
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    for rec in _read_jsonl(path):
        cache[_cache_key(rec["query_id"], rec["chunk_id"])] = rec
    return cache


def _save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(path, list(cache.values()))


# --------------------------------------------------------------------------
# Anthropic judge 클라이언트 (전용 env 키, backoff 재시도)
# --------------------------------------------------------------------------
def _read_judge_api_key() -> str | None:
    """전용 env 에서만 키를 읽는다. 전역 ANTHROPIC_API_KEY 는 절대 건드리지 않는다."""
    key = os.environ.get(JUDGE_API_KEY_ENV)
    return key.strip() if key else None


class AnthropicJudge:
    """Claude 호출 래퍼. sync + backoff 재시도.

    ``anthropic`` 은 **지연 import** 한다 — 이 모듈을 import 하거나 sample/kappa 를
    돌리는 데에는 SDK 가 필요 없게 하기 위함이다.
    """

    def __init__(self, api_key: str, model: str):
        try:
            import anthropic  # 지연 import
        except ImportError as exc:  # pragma: no cover
            _die(
                "anthropic SDK 가 설치되지 않았습니다. `pip install -e \".[eval]\"` "
                f"또는 `pip install anthropic`. (원인: {exc})"
            )
        self._anthropic = anthropic
        # 키를 명시 전달 — 전역 ANTHROPIC_API_KEY 환경변수에 의존하지 않는다.
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(
        self, system: str, user: str, max_tokens: int, model: str | None = None
    ) -> str:
        """system + user 로 1턴 메시지를 보내고 텍스트를 반환한다.

        RateLimitError / 5xx / 연결·타임아웃 오류는 지수 backoff 로 재시도한다.
        그 외 오류(인증 등)는 즉시 올린다.
        """
        a = self._anthropic
        transient = (
            a.RateLimitError,
            a.APITimeoutError,
            a.APIConnectionError,
            a.InternalServerError,
        )
        rng = random.Random(SEED)  # jitter (재현성)
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.messages.create(
                    model=model or self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                parts = [
                    blk.text for blk in resp.content if getattr(blk, "type", "") == "text"
                ]
                return "".join(parts).strip()
            except transient as exc:  # noqa: PERF203
                last_exc = exc
                wait = BACKOFF_BASE * (2**attempt) + rng.uniform(0, 1)
                logger.warning(
                    "transient API error (attempt %d/%d): %s — %.1fs 후 재시도",
                    attempt + 1, MAX_RETRIES, type(exc).__name__, wait,
                )
                time.sleep(wait)
            except a.APIStatusError as exc:  # 5xx 만 재시도, 4xx 는 즉시 올림
                status = getattr(exc, "status_code", None)
                if status is not None and 500 <= status < 600:
                    last_exc = exc
                    wait = BACKOFF_BASE * (2**attempt) + rng.uniform(0, 1)
                    logger.warning(
                        "API %s (attempt %d/%d) — %.1fs 후 재시도",
                        status, attempt + 1, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(
            f"Anthropic 호출이 {MAX_RETRIES}회 재시도 후에도 실패: "
            f"{type(last_exc).__name__}: {last_exc}"
        )


def _parse_label(text: str) -> tuple[int, bool]:
    """판정 텍스트에서 LABEL: 0|1 을 파싱한다.

    Returns:
        (label, parsed_ok). 파싱 실패면 (0, False) — 보수적으로 0.
    """
    matches = LABEL_RE.findall(text)
    if not matches:
        return 0, False
    return int(matches[-1]), True  # 마지막 LABEL: 줄을 채택


# --------------------------------------------------------------------------
# Qdrant read-only helpers
# --------------------------------------------------------------------------
def _make_client():
    from qdrant_client import QdrantClient  # 지연 import

    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _ensure_collection(client, collection: str) -> None:
    try:
        client.get_collection(collection)
    except Exception as exc:  # noqa: BLE001
        _die(
            f"Qdrant collection not found or unreachable: {collection!r}. "
            f"cause: {type(exc).__name__}: {exc}"
        )


def _sample_seed_chunks(client, collection: str, n: int) -> list[dict]:
    """scroll 로 read-only 표본을 뽑아 chunk type/파일 고루 stratify 한다.

    Returns:
        [{chunk_id, text, type, file_path}, ...]  (최대 n 개)
    """
    rng = random.Random(SEED)
    cap = max(n * 20, 500)
    candidates: list[dict] = []
    offset = None
    while len(candidates) < cap:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            pl = p.payload or {}
            text = pl.get("text") or ""
            if not text.strip():
                continue
            candidates.append(
                {
                    "chunk_id": str(pl.get("chunk_id") or p.id),
                    "text": text,
                    "type": pl.get("type") or "UNKNOWN",
                    "file_path": pl.get("file_path") or "",
                }
            )
        if offset is None:
            break

    if not candidates:
        _die(f"collection {collection!r} 에서 표본을 뽑지 못했습니다(비어있음?).")

    # type 별 버킷 → 시드 셔플 → 라운드로빈, 파일당 soft 상한으로 파일 다양성 확보.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        buckets[c["type"]].append(c)
    for t in buckets:
        rng.shuffle(buckets[t])

    order = list(buckets.keys())
    rng.shuffle(order)
    idxs = {t: 0 for t in buckets}
    file_count: Counter = Counter()
    selected: list[dict] = []

    def _drain(respect_cap: bool) -> None:
        progressed = True
        while len(selected) < n and progressed:
            progressed = False
            for t in order:
                if len(selected) >= n:
                    break
                i = idxs[t]
                while i < len(buckets[t]):
                    c = buckets[t][i]
                    i += 1
                    if not respect_cap or file_count[c["file_path"]] < SEED_FILE_CAP:
                        selected.append(c)
                        file_count[c["file_path"]] += 1
                        progressed = True
                        break
                idxs[t] = i

    _drain(respect_cap=True)
    if len(selected) < n:  # 파일 상한이 너무 빡빡하면 남은 것으로 채운다
        _drain(respect_cap=False)
    return selected[:n]


def _retrieve_texts(client, collection: str, chunk_ids: list[str]) -> dict[str, str]:
    """chunk_id → text (read-only client.retrieve by id)."""
    if not chunk_ids:
        return {}
    found = client.retrieve(
        collection_name=collection,
        ids=chunk_ids,
        with_payload=True,
        with_vectors=False,
    )
    out: dict[str, str] = {}
    for p in found:
        pl = p.payload or {}
        out[str(pl.get("chunk_id") or p.id)] = pl.get("text") or ""
    return out


# --------------------------------------------------------------------------
# provenance 사이드카
# --------------------------------------------------------------------------
def _provenance_path(gt_path: Path) -> Path:
    return gt_path.with_name("ground_truth.provenance.json")


def _load_provenance(gt_path: Path) -> dict:
    p = _provenance_path(gt_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_provenance(gt_path: Path, prov: dict) -> None:
    _provenance_path(gt_path).write_text(
        json.dumps(prov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _prompt_meta() -> dict:
    return {
        "gen_model": GEN_MODEL,
        "judge_model": JUDGE_MODEL,
        "gen_prompt_version": GEN_PROMPT_VERSION,
        "gen_prompt_hash": _sha8(GEN_SYSTEM + "\n" + GEN_USER_TEMPLATE),
        "rubric_prompt_version": RUBRIC_PROMPT_VERSION,
        "rubric_prompt_hash": _sha8(RUBRIC_SYSTEM + "\n" + JUDGE_USER_TEMPLATE),
        "gen_n": GEN_N,
        "pool_depth": POOL_DEPTH,
        "seed": SEED,
        "queries_are_synthetic": True,
        "note": "queries are synthetic (auto-generated)",
    }


# ==========================================================================
# 모드 1: gen-queries
# ==========================================================================
def cmd_gen_queries(args: argparse.Namespace) -> None:
    key = _read_judge_api_key()
    if not key:
        _die(
            f"{JUDGE_API_KEY_ENV} 가 설정되지 않았습니다. "
            f"`export {JUDGE_API_KEY_ENV}=...` 설정 후 실행하세요.",
            code=1,
        )

    gt_path = Path(args.gt)
    n = args.n
    oversample = max(n, int(round(n * GEN_OVERSAMPLE)))

    client = _make_client()
    _ensure_collection(client, args.collection)
    logger.info("seed 청크 표본 추출 중 (target=%d, oversample=%d)...", n, oversample)
    seeds = _sample_seed_chunks(client, args.collection, oversample)
    logger.info("seed %d 개 추출. 쿼리 생성 시작.", len(seeds))

    judge = AnthropicJudge(api_key=key, model=GEN_MODEL)

    def _gen_one(seed: dict) -> str | None:
        user = GEN_USER_TEMPLATE.format(chunk_text=seed["text"][:4000])
        try:
            out = judge.complete(GEN_SYSTEM, user, GEN_MAX_TOKENS, model=GEN_MODEL)
        except Exception as exc:  # noqa: BLE001 - seed 단위 격리
            logger.warning("쿼리 생성 실패(chunk_id=%s): %s", seed["chunk_id"], exc)
            return None
        # 모델이 여러 줄/따옴표로 감싸면 첫 비어있지 않은 줄만 취한다.
        line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
        return line.strip().strip('"').strip("'") or None

    raw: list[str] = []
    with ThreadPoolExecutor(max_workers=THREADPOOL_MAX) as pool:
        for q in pool.map(_gen_one, seeds):
            if q:
                raw.append(q)

    # 근접 중복 제거(정규화 문자열 기준) 후 GEN_N 으로 절단.
    seen: set[str] = set()
    deduped: list[str] = []
    for q in raw:
        norm = _normalize_query(q)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(q)
    deduped = deduped[:n]

    if not deduped:
        _die("생성된 쿼리가 없습니다(모두 실패/중복).")

    # seed 청크를 relevant 로 박지 않는다 — 라벨링이 독립 수행한다.
    records = [
        {"query_id": f"q{i:03d}", "query": q, "relevant_chunk_ids": []}
        for i, q in enumerate(deduped, 1)
    ]
    _write_jsonl_atomic(gt_path, records)

    prov = _load_provenance(gt_path)
    prov.update(_prompt_meta())
    prov.update(
        {
            "stage_gen_queries": {
                "timestamp": _now_iso(),
                "collection": args.collection,
                "seeds_sampled": len(seeds),
                "queries_generated": len(deduped),
                "gen_n": n,
            },
            "query_count": len(records),
        }
    )
    _save_provenance(gt_path, prov)

    print(f"[gen-queries] {len(records)} 쿼리 생성 → {gt_path}")
    print(f"[gen-queries] provenance → {_provenance_path(gt_path)}")
    for r in records[:3]:
        print(f"  {r['query_id']}: {r['query']}")


# ==========================================================================
# 모드 2: label
# ==========================================================================
def _pool_for_query(retriever, configs, query: str, depth: int, cap: int | None) -> list[str]:
    """ablation 의 모든 config top-depth chunk_id 합집합(순서 보존 dedup)."""
    seen: dict[str, None] = {}
    for cfg in configs:
        try:
            pairs = retriever.search_one(query, cfg, depth)
        except Exception as exc:  # noqa: BLE001 - config 단위 격리
            logger.warning("pooling 검색 실패 (config=%s): %s", cfg.name, exc)
            continue
        for cid, _score in pairs:
            seen.setdefault(cid, None)
    pool = list(seen.keys())
    if cap is not None:
        pool = pool[:cap]
    return pool


def cmd_label(args: argparse.Namespace) -> None:
    key = _read_judge_api_key()
    if not key:
        _die(
            f"{JUDGE_API_KEY_ENV} 가 설정되지 않았습니다. "
            f"`export {JUDGE_API_KEY_ENV}=...` 설정 후 실행하세요.",
            code=1,
        )

    gt_path = Path(args.gt)
    records = _read_jsonl(gt_path)
    cache_path = _cache_path(gt_path)
    cache = _load_cache(cache_path)

    # 검색 재사용 (production 로직 import 만, 수정 없음)
    from ragent.evaluation.retrieval_runner import (
        EvalRetriever,
        default_ablation_matrix,
    )

    retriever = EvalRetriever()
    _ensure_collection(retriever.client, args.collection)
    configs = default_ablation_matrix(args.collection)

    judge = AnthropicJudge(api_key=key, model=JUDGE_MODEL)

    todo = records if args.limit_queries is None else records[: args.limit_queries]
    total_judgments = 0

    for rec in todo:
        qid = rec.get("query_id")
        query = rec.get("query") or ""
        if not query.strip():
            logger.warning("빈 query 건너뜀: query_id=%s", qid)
            continue

        pool = _pool_for_query(
            retriever, configs, query, POOL_DEPTH, args.pool_cap
        )
        texts = _retrieve_texts(retriever.client, args.collection, pool)
        logger.info("query_id=%s pool=%d chunks", qid, len(pool))

        # (query, chunk) 독립 판정. 캐시 히트면 재호출 안 함.
        def _judge_one(cid: str) -> dict:
            ck = _cache_key(qid, cid)
            if ck in cache:
                return cache[ck]
            text = texts.get(cid, "")
            user = JUDGE_USER_TEMPLATE.format(query=query, chunk_text=text[:6000])
            try:
                out = judge.complete(
                    RUBRIC_SYSTEM, user, JUDGE_MAX_TOKENS, model=JUDGE_MODEL
                )
                label, ok = _parse_label(out)
                if not ok:
                    logger.warning(
                        "LABEL 파싱 실패 → 0 (query_id=%s chunk_id=%s)", qid, cid
                    )
                rationale = out
            except Exception as exc:  # noqa: BLE001 - 판정 단위 격리
                logger.warning("판정 실패 → 0 (query_id=%s chunk_id=%s): %s", qid, cid, exc)
                label, rationale = 0, f"[error] {type(exc).__name__}: {exc}"
            return {
                "query_id": qid,
                "chunk_id": cid,
                "label": label,
                "rationale": rationale,
                "query": query,
                "text": (texts.get(cid, "") or "")[:600],  # sample snippet 용 보관
            }

        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=THREADPOOL_MAX) as tp:
            for res in tp.map(_judge_one, pool):
                results.append(res)
                cache[_cache_key(res["query_id"], res["chunk_id"])] = res
                total_judgments += 1
        # 매 쿼리 후 캐시 저장(중단 시에도 진척 보존)
        _save_cache(cache_path, cache)

        rec["relevant_chunk_ids"] = [r["chunk_id"] for r in results if r["label"] == 1]

    _write_jsonl_atomic(gt_path, records)

    prov = _load_provenance(gt_path)
    prov.update(_prompt_meta())
    prov.update(
        {
            "stage_label": {
                "timestamp": _now_iso(),
                "collection": args.collection,
                "queries_labeled": len(todo),
                "total_judgments": total_judgments,
                "pool_depth": POOL_DEPTH,
            }
        }
    )
    _save_provenance(gt_path, prov)

    print(f"[label] {len(todo)} 쿼리 라벨링, 총 {total_judgments} 판정 → {gt_path}")
    print(f"[label] 캐시 → {cache_path}")
    print(f"[label] provenance → {_provenance_path(gt_path)}")


# ==========================================================================
# 모드 3: sample (stratified κ 표본)
# ==========================================================================
def _stratified_sample(cache_records: list[dict], n: int) -> list[dict]:
    """LLM-pos/neg 균형(각 n/2) + 쿼리에 고루 분포. seed 고정.

    각 라벨 클래스를 query_id 로 그룹핑해 라운드로빈으로 뽑아 쿼리 편중을 줄인다.
    한쪽 클래스가 모자라면 다른 클래스에서 채운다.
    """
    rng = random.Random(SEED)
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for r in cache_records:
        by_label.get(int(r["label"]), by_label[0]).append(r)

    def _pick(records: list[dict], k: int) -> list[dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            groups[r["query_id"]].append(r)
        qids = list(groups.keys())
        rng.shuffle(qids)
        for q in qids:
            rng.shuffle(groups[q])
        picked: list[dict] = []
        idxs = {q: 0 for q in qids}
        progressed = True
        while len(picked) < k and progressed:
            progressed = False
            for q in qids:
                if len(picked) >= k:
                    break
                if idxs[q] < len(groups[q]):
                    picked.append(groups[q][idxs[q]])
                    idxs[q] += 1
                    progressed = True
        return picked

    half = n // 2
    pos = _pick(by_label[1], half)
    neg = _pick(by_label[0], n - half)
    # 한쪽 부족 시 반대편에서 보충
    if len(pos) + len(neg) < n:
        deficit = n - (len(pos) + len(neg))
        chosen = {_cache_key(r["query_id"], r["chunk_id"]) for r in pos + neg}
        extra_src = by_label[0] if len(pos) >= half else by_label[1]
        extra = [
            r for r in extra_src
            if _cache_key(r["query_id"], r["chunk_id"]) not in chosen
        ]
        rng.shuffle(extra)
        neg.extend(extra[:deficit])

    sample = pos + neg
    rng.shuffle(sample)
    return sample


def cmd_sample(args: argparse.Namespace) -> None:
    gt_path = Path(args.gt)
    cache_path = Path(args.cache) if args.cache else _cache_path(gt_path)
    if not cache_path.exists():
        _die(f"라벨 캐시가 없습니다: {cache_path} (먼저 label 모드를 실행하세요)")

    cache_records = _read_jsonl(cache_path)
    if not cache_records:
        _die(f"라벨 캐시가 비어있습니다: {cache_path}")

    sample = _stratified_sample(cache_records, args.n)

    out_path = Path(args.out)
    rows = [
        {
            "query_id": r["query_id"],
            "query": r.get("query", ""),
            "chunk_id": r["chunk_id"],
            "chunk_snippet": (r.get("text", "") or "")[:400],
            "llm_label": int(r["label"]),
            "llm_rationale": r.get("rationale", ""),
            "human_label": None,
        }
        for r in sample
    ]
    _write_jsonl_atomic(out_path, rows)

    n_pos = sum(1 for r in rows if r["llm_label"] == 1)
    n_neg = len(rows) - n_pos
    print(f"[sample] {len(rows)} 표본 (pos={n_pos}, neg={n_neg}) → {out_path}")
    print("[sample] human_label 을 채운 뒤 `kappa` 모드를 실행하세요.")


# ==========================================================================
# 모드 4: kappa
# ==========================================================================
def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    """Cohen's κ 직접 계산. pairs = [(rater_a, rater_b), ...] (0/1)."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    # 주변 확률
    a1 = sum(1 for a, _ in pairs if a == 1) / n
    b1 = sum(1 for _, b in pairs if b == 1) / n
    pe = a1 * b1 + (1 - a1) * (1 - b1)
    if pe == 1.0:  # 완전 일치 주변분포 → κ 정의불가, 관측일치로 대체
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def cmd_kappa(args: argparse.Namespace) -> None:
    sample_path = Path(args.sample)
    rows = _read_jsonl(sample_path)
    if not rows:
        _die(f"표본이 비어있습니다: {sample_path}")

    pairs: list[tuple[int, int]] = []
    n_null = 0
    for r in rows:
        hl = r.get("human_label")
        if hl is None:
            n_null += 1
            continue
        pairs.append((int(r["llm_label"]), int(hl)))

    if n_null:
        print(f"[kappa] 경고: human_label 이 비어있는 표본 {n_null}건 — 제외하고 계산합니다.")
    if not pairs:
        _die("human_label 이 채워진 표본이 없습니다.")

    # 2x2 혼동표 (행=llm, 열=human)
    cm = Counter()
    for llm, hum in pairs:
        cm[(llm, hum)] += 1
    raw_agreement = sum(1 for a, b in pairs if a == b) / len(pairs)
    kappa = cohen_kappa(pairs)

    print(f"\n[kappa] n={len(pairs)} (null 제외 {n_null})")
    print(f"[kappa] Cohen's κ      = {kappa:.4f}")
    print(f"[kappa] raw agreement  = {raw_agreement * 100:.1f}%")
    print("\n[kappa] 2x2 혼동표 (행=LLM, 열=Human)")
    print("            human=0   human=1")
    print(f"  llm=0     {cm[(0, 0)]:>7}   {cm[(0, 1)]:>7}")
    print(f"  llm=1     {cm[(1, 0)]:>7}   {cm[(1, 1)]:>7}")


# ==========================================================================
# CLI
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragent.evaluation.gt_labeling",
        description="GT 라벨링 풀 자동 파이프라인 (gen-queries / label / sample / kappa)",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("gen-queries", help="코퍼스에서 backward-reference 쿼리 생성")
    g.add_argument("--collection", default=DEFAULT_COLLECTION)
    g.add_argument("--gt", default=str(DEFAULT_GT))
    g.add_argument("--n", type=int, default=GEN_N, help="최종 쿼리 수")
    g.set_defaults(func=cmd_gen_queries)

    lb = sub.add_parser("label", help="pool 의 (query, chunk) relevance 전수 라벨링")
    lb.add_argument("--collection", default=DEFAULT_COLLECTION)
    lb.add_argument("--gt", default=str(DEFAULT_GT))
    lb.add_argument("--limit-queries", type=int, default=None, help="앞 N개 쿼리만(스모크)")
    lb.add_argument("--pool-cap", type=int, default=None, help="쿼리당 pool 상한(스모크)")
    lb.set_defaults(func=cmd_label)

    sp = sub.add_parser("sample", help="라벨 캐시에서 stratified κ 표본 추출")
    sp.add_argument("--gt", default=str(DEFAULT_GT))
    sp.add_argument("--cache", default=None, help="라벨 캐시 경로(기본: GT 옆 .gt_cache)")
    sp.add_argument("--out", default=str(DEFAULT_SAMPLE))
    sp.add_argument("--n", type=int, default=KAPPA_SAMPLE_N)
    sp.set_defaults(func=cmd_sample)

    kp = sub.add_parser("kappa", help="human_label 채운 표본으로 Cohen's κ 계산")
    kp.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    kp.set_defaults(func=cmd_kappa)

    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
