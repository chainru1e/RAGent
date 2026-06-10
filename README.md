# RAGent

코딩 에이전트(Claude Code · Codex · Windsurf)의 대화와 코드 변경을 hook으로 자동 수집하여 로컬 벡터 데이터베이스(Qdrant)에 색인하고, 사용자가 새 프롬프트를 제출할 때 관련된 과거 컨텍스트를 검색해 에이전트에 주입하는 로컬 RAG(Retrieval-Augmented Generation) 시스템입니다.

임베딩·재랭킹·쿼리 변환·인텐트 분류·맥락 생성에 쓰이는 LLM을 포함한 모든 구성요소가 로컬에서 동작합니다. 외부 API 키가 필요 없으며, 무거운 모델은 백그라운드 서버 프로세스에 상주시켜 hook 호출마다 다시 로드하지 않습니다.

---

## 아키텍처

RAGent는 **상주 서버 3종**과, hook 시점마다 단명(short-lived)으로 실행되는 **클라이언트**로 나뉩니다.

```
┌──────────────────────────────────────────────────────────────┐
│  Coding Agent (Claude Code / Codex / Windsurf)                │
│                                                              │
│   prompt 제출                          응답 완료              │
│      │                                    │                  │
│      ▼ (hook)                             ▼ (hook)           │
│  python -m ragent --adapter <name>   python -m ragent ...    │
└──────┬───────────────────────────────────┬──────────────────┘
       │ POST /search                       │ POST /save
       │ (timeout 30s)                      │ (timeout 5s)
       ▼                                    ▼
┌──────────────────────────────────────────────────────────────┐
│  RAGent server  (127.0.0.1:8765)                             │
│   /search → 검색 후 컨텍스트 반환                              │
│   /save   → 큐에 적재(202 즉시 응답) → save worker가 백그라운드 색인 │
└──────┬───────────────────────────────────┬──────────────────┘
       │                                    │
       ▼                                    ▼
┌──────────────────┐              ┌──────────────────────────┐
│ LLM server       │              │ Qdrant (Docker)          │
│ 127.0.0.1:8000   │              │ 127.0.0.1:6333           │
│ llama-cpp /v1    │              │ container: ragent-qdrant │
└──────────────────┘              └──────────────────────────┘
```

세 서버는 `python -m ragent.launcher`가 함께 기동·종료합니다.

**데이터 흐름**

1. **프롬프트 제출 시** — 클라이언트가 RAGent 서버 `POST /search`를 호출하고(핸들러 timeout 30s), 검색된 컨텍스트를 `hookSpecificOutput.additionalContext`로 표준출력에 내보내 에이전트 프롬프트에 주입합니다.
2. **응답 완료 시** — 클라이언트가 `POST /save`를 호출하고(timeout 5s), 서버는 요청을 큐에 적재한 뒤 즉시 `202`로 응답합니다. 단일 save worker 스레드가 큐를 직렬로 비우며 백그라운드에서 색인합니다(임베딩 모델·로컬 LLM에 대한 동시 접근 방지).

---

## 멀티 에이전트 어댑터

호스트 에이전트별 hook 스키마 차이는 어댑터가 흡수합니다. 어댑터는 `ragent/adapters/__init__.py`의 `ADAPTER_REGISTRY`에 등록됩니다.

| 어댑터 (`--adapter`) | hook 설치 위치 | 이벤트 매핑 (정규화) | 파서 |
|---|---|---|---|
| `claude_code` (기본) | `~/.claude/settings.json` | `UserPromptSubmit`→prompt, `Stop`→response | `ClaudeCodeParser` |
| `codex` | `~/.codex/hooks.json` | `UserPromptSubmit`→prompt, `Stop`→response | `CodexParser` |
| `windsurf` | `~/.codeium/windsurf/hooks.json` | `pre_user_prompt`→prompt, `post_cascade_response_with_transcript`→response | `WindsurfParser` |

**어댑터 선택 우선순위** (`get_adapter`): hook 명령에 박힌 `--adapter` 인자 → 환경변수 `RAGENT_ADAPTER` → 기본값 `claude_code`.

**현재 한계 (코드 기준):** Codex와 Windsurf의 파일 쓰기는 Claude Code의 `[Write]` tool 컨벤션과 호환되지 않아, 두 어댑터에서는 코드 청크 인덱싱이 발화하지 않고 텍스트 컨텍스트 인덱싱만 동작합니다.

---

## 인덱싱 파이프라인

`POST /save` 요청을 받은 save worker가 transcript의 최근 1턴을 다음 순서로 처리합니다 (`RAGentServer._process_save` → `index_turn`).

1. **파싱** — `BaseParser.parse_last_turn()`이 transcript를 끝에서부터 역순으로 읽어 사용자 텍스트(`[text]`)를 시작점으로 하는 최근 1턴을 추출합니다.
2. **청킹** — `Chunker.process_turn()`이 턴을 ① 문맥(부모) 청크 1개와 ② Write/Edit/MultiEdit tool 사용에서 추출한 코드(자식) 청크들로 나눕니다. 코드 청크는 `astchunk`로 AST 단위(함수/클래스 등)로 다시 쪼갭니다(지원 언어: Python, Java, C#, TypeScript, JavaScript). 부모-자식은 결정론적 UUID5 ID와 `parent_id`로 연결됩니다.
3. **인텐트 분류** — `IntentClassifier`가 로컬 LLM으로 문맥을 4개 카테고리(`CODE_GENERATION`, `CODE_REFACTORING`, `CODE_DEBUGGING`, `SIMPLE_QUESTION`) 중 하나로 분류합니다.
4. **Contextual enrichment** *(선택)* — 아래 [Contextual Retrieval](#contextual-retrieval) 참조.
5. **임베딩** — `HybridEmbedding`이 dense + sparse 벡터를 생성합니다. dense는 로컬 임베딩 모델(Qwen3-Embedding-0.6B GGUF, llama-cpp)로 만들고 MRL 방식으로 `dense_short`(256차원)·`dense_long`(1024차원) 두 벡터를 둡니다. sparse는 `Qdrant/bm25`(fastembed)로 만듭니다.
6. **저장** — `QdrantStorage.add_points_batch()`로 upsert합니다.

**컬렉션 구조:** 컬렉션 이름은 transcript 경로의 부모 디렉터리명으로 정해집니다(프로젝트별 1컬렉션). 각 컬렉션은 named vector 3종을 가집니다 — `dense_short`(256, Cosine), `dense_long`(1024, Cosine), `sparse`.

---

## Contextual Retrieval

코드 청크 앞에 LLM이 생성한 짧은 맥락 문장(prefix)을 붙여 검색 품질을 높이는 전처리입니다.

- 활성화: `ENABLE_CONTEXTUAL_RETRIEVAL`(기본 `True`).
- 문서 단위: `CONTEXTUAL_DOC_GRANULARITY` — `"turn"` 또는 `"file"`(기본 `"file"`).
  - `turn`: 청크가 나온 턴 전체(`serialize_turn`)를 문서로 사용.
  - `file`: 청크가 나온 파일 전체를 문서로 사용. 문서 선택 우선순위는 ① 동일 턴의 Write 본문 → ② 세션 전체에서 모은 최신 Write 스냅샷(`build_file_snapshots`) → ③ `serialize_turn` 폴백.
- 큰 파일은 AST skeleton(시그니처 + docstring만 남김)으로 압축합니다(현재 Python만 구현, 그 외 언어/파싱 실패 시 head 기준 truncate). 문서 길이 상한은 `MAX_FILE_DOC_CHARS`(6000자).
- 생성된 prefix는 **임베딩 입력에만** `"prefix\n\npayload"` 형태로 합쳐지고, Qdrant에 저장되는 `payload`와 검색 결과로 노출되는 본문은 원본을 유지합니다.

---

## 검색 파이프라인

`POST /search` 요청을 `Retriever.retrieve()`가 처리합니다.

1. **쿼리 변환** — `QueryTransformer`가 로컬 LLM으로 질문을 분해·재작성하고 키워드를 확장합니다(`json_repair`로 응답 파싱, 실패 시 원문 쿼리로 폴백).
2. **하이브리드 검색** — `QdrantStorage.staged_hybrid_search()`가 `dense_short`로 후보를 빠르게 추린 뒤 `dense_long`으로 정밀 재검색하고, sparse 검색과 함께 RRF(Reciprocal Rank Fusion)로 융합합니다.
3. **재랭킹** — `Reranker`가 Cross-Encoder(`BAAI/bge-reranker-v2-m3`)로 재점수화합니다.
4. **컷오프** — `static_cutoff`(점수 0.3 미만 제거) → `dynamic_cutoff`(점수 낙폭 0.1 초과 지점에서 절단).
5. 살아남은 청크를 `<context><document>...</document></context>` XML로 포맷해 반환합니다.

---

## 로컬 LLM 서버

`unsloth/Qwen3.5-9B-GGUF`(`Qwen3.5-9B-Q4_K_M.gguf`)를 Hugging Face Hub에서 자동 다운로드하여, `llama-cpp-python`의 OpenAI 호환 서버 + `uvicorn`으로 `http://127.0.0.1:8000/v1`에 띄웁니다(`ragent/llm_server.py`). 인텐트 분류·쿼리 변환·Contextual enrichment 세 곳에서 이 서버를 호출합니다.

기동 시 하드웨어를 스캔하여 `n_gpu_layers`·`n_ctx`·`n_threads`를 자동 산정합니다.

| 환경 | n_gpu_layers | n_ctx |
|---|---|---|
| Apple Silicon | -1 (전체, Metal) | 4096 |
| VRAM ≥ 7.5 GB | -1 (전체) | 4096 |
| VRAM ≥ 5.5 GB | 20 | 4096 |
| VRAM ≥ 3.5 GB | 10 | 4096 |
| RAM ≥ 15.5 GB (GPU 미사용) | 0 | 4096 |
| 그 외 | 0 | 2048 |

---

## 설치 및 실행

### 요구 사항

- **Python 3.10 이상**
- **Docker (설치 + 실행 중) — 필수.** Qdrant를 Docker 컨테이너로 띄웁니다. Docker가 설치되어 있지 않거나 실행 중이 아니면 launcher가 `RuntimeError`로 중단됩니다.
- 호스트 코딩 에이전트(Claude Code / Codex / Windsurf) 중 하나

### 1. 의존성 설치

```bash
cd RAGent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

개발 의존성(pytest, pyinstaller)까지 설치하려면:

```bash
pip install -e ".[dev]"
```

> **참고:** `pyproject.toml`의 런타임 의존성은 `astchunk`, `peft`, `qdrant-client`, `sentence_transformers`, `llama-cpp-python[server]`, `uvicorn`, `openai`, `fastembed`, `json-repair`입니다. 코드는 이 외에 `psutil`(하드웨어 스캔)과 `requests`(launcher의 헬스체크/폴링)도 사용하지만 현재 이 둘은 의존성 목록에 포함되어 있지 않습니다. 누락 시 별도로 설치해야 합니다.

### 2. hook 등록

```bash
python install.py                       # 기본: claude_code
python install.py --adapter codex       # 또는 codex / windsurf
```

선택한 어댑터의 hook 설정 파일(위 [어댑터 표](#멀티-에이전트-어댑터) 참조)에 RAGent hook을 병합합니다. 기존 RAGent hook은 식별 후 제거하고 새로 등록하므로 여러 번 실행해도 중복되지 않습니다.

### 3. 서버 기동

hook이 동작하려면 서버 3종이 떠 있어야 합니다.

```bash
python -m ragent.launcher
```

launcher는 4단계로 진행합니다: `[1/4]` Qdrant 컨테이너 시작 → `[2/4]` 모델 준비 → `[3/4]` LLM 서버 → `[4/4]` RAGent 서버. 이후 작업 상태(검색/저장 진행 상황)를 콘솔에 요약 출력하며, `Ctrl+C`로 전체를 종료합니다.

**최초 실행 시** 모델(LLM·임베딩·리랭커)을 Hugging Face에서 내려받고 `qdrant/qdrant` Docker 이미지를 pull하므로 시간이 걸립니다. 이미 캐시에 있으면 다운로드를 건너뜁니다.

---

## frozen(exe) 모드

RAGent는 소스 모드(`python -m ...`)와 PyInstaller로 빌드된 frozen 실행파일 모드 양쪽에서 동작하도록 설계되어 있습니다. frozen 모드에서는 역할별로 별도 exe(`Launcher`, `Server`, `LLMServer`, hook 진입점 `RAGent-hook`)를 사용하며, 이 경우 `install.py`가 등록하는 hook 명령도 `-m ragent` 대신 `RAGent-hook` exe를 가리킵니다. `pyinstaller>=6.0`은 `pyproject.toml`의 dev 의존성에 포함되어 있으나, `.spec` 파일과 빌드 스크립트는 저장소에 커밋되어 있지 않습니다.

---

## 파일 구조

```
RAGent/
├── pyproject.toml                  # 패키지 메타데이터 · 의존성
├── install.py                      # 어댑터 hook 등록 (--adapter)
└── ragent/
    ├── __main__.py                 # `python -m ragent` 진입점 (hook 클라이언트)
    ├── main.py                     # stdin JSON 파싱 → 어댑터 dispatch
    ├── launcher.py                 # 서버 3종 기동·종료, 모델 프리페치, 상태 출력
    ├── server.py                   # RAGent 서버 (HTTP, save 큐/worker, /search)
    ├── client.py                   # 서버 호출용 HTTP 클라이언트 (post_json)
    ├── llm_server.py               # 로컬 LLM 서버 (llama-cpp + uvicorn)
    ├── llm_client.py               # OpenAI 호환 LLM 클라이언트
    ├── vectordb_manager.py         # Qdrant Docker 컨테이너 수명관리 (QdrantManager)
    ├── vectordb_client.py          # Qdrant 검색/upsert (QdrantStorage)
    ├── config.py                   # 경로·포트·모델·상수
    ├── logging_config.py           # 컴포넌트별 로그 파일 구성
    ├── utils.py                    # 하드웨어 스캔, pause_if_frozen
    ├── adapters/
    │   ├── base.py                 # BaseAdapter (dispatch, build_hook_command)
    │   ├── claude_code.py          # ClaudeCodeAdapter
    │   ├── codex.py                # CodexAdapter
    │   └── windsurf.py             # WindsurfAdapter
    ├── handlers/
    │   ├── user_prompt_submit.py   # prompt 이벤트 → /search → 컨텍스트 주입
    │   └── stop.py                 # response 이벤트 → /save
    ├── modules/
    │   ├── parsing_modules.py      # BaseParser + 어댑터별 파서
    │   ├── chunking_modules.py     # Chunker (문맥/코드 청크, AST 분할)
    │   ├── intent_classifying_modules.py  # IntentClassifier
    │   ├── embedding_modules.py    # HybridEmbedding (dense + sparse)
    │   ├── contextual_retrieval_modules.py # ContextualEnricher
    │   ├── code_skeleton.py        # 파일 문서용 AST skeleton 추출
    │   ├── indexing_modules.py     # index_turn (오케스트레이션 + 신뢰성)
    │   └── retrieval_modules.py    # Retriever, Reranker, QueryTransformer, cutoff
    └── models/
        ├── chunk.py                # Chunk, ChunkMetaData
        ├── intent.py              # IntentCategory, ClassificationResult
        ├── parsed_message.py       # NormalizedMessage, Block
        ├── transformed_query.py    # TransformedQuery
        └── vector.py              # HybridVector
```

---

## 설정 레퍼런스 (`config.py`)

| 상수 | 값 | 설명 |
|---|---|---|
| `BASE_DIR` | `~/.ragent/` | 루트 디렉터리 |
| `QDRANT_DIR` | `~/.ragent/qdrant_storage/` | Qdrant 컨테이너 볼륨 마운트 대상 |
| `LOG_DIR` | `~/.ragent/logs/` | 로그 디렉터리 |
| `FAILED_CHUNKS_FILE` | `~/.ragent/failed_chunks.jsonl` | 영구 실패 색인 기록 |
| `RAGENT_SERVER_PORT` | `8765` | RAGent 서버 포트 |
| `QDRANT_PORT` | `6333` | Qdrant 포트 |
| `LLM_SERVER_PORT` | `8000` | LLM 서버 포트 |
| `LLM_REPO_ID` / `LLM_FILENAME` | `unsloth/Qwen3.5-9B-GGUF` / `Qwen3.5-9B-Q4_K_M.gguf` | 로컬 LLM |
| `DENSE_EMBEDDING_REPO_ID` / `DENSE_EMBEDDING_FILENAME` | `Qwen/Qwen3-Embedding-0.6B-GGUF` / `Qwen3-Embedding-0.6B-Q8_0.gguf` | dense 임베딩 |
| `SPARSE_EMBEDDING_MODEL` | `Qdrant/bm25` | sparse 임베딩 |
| `RERANKING_MODEL` | `BAAI/bge-reranker-v2-m3` | 재랭킹 Cross-Encoder |
| `SHORT_DENSE_SIZE` / `LONG_DENSE_SIZE` | `256` / `1024` | MRL dense 벡터 차원 |
| `MAX_CHUNK_SIZE` | `1000` | 청크당 최대 문자 수 |
| `MAX_FILE_DOC_CHARS` | `6000` | 파일 문서 길이 상한 |
| `RERANK_BATCH_SIZE` | `8` | 재랭킹 배치 크기 |
| `RERANK_MAX_LENGTH` | `None` | 재랭킹 입력 길이 캡(미적용) |
| `ENABLE_CONTEXTUAL_RETRIEVAL` | `True` | Contextual Retrieval 활성화 |
| `CONTEXTUAL_DOC_GRANULARITY` | `"file"` | 문서 단위(`turn`/`file`) |

로그 파일 경로: `LOG_FILE`(ragent.log), `RAGENT_SERVER_LOG_FILE`, `LLM_LOG_FILE`, `VECTORDB_LOG_FILE`, 그리고 launcher가 자식 서버 stdout/stderr를 원본 캡처하는 `RAGENT_SERVER_CONSOLE_LOG`·`LLM_SERVER_CONSOLE_LOG`.

`ensure_dirs()`가 `QDRANT_DIR`과 `LOG_DIR`을 생성합니다.

---

## 데이터 저장 위치

```
~/.ragent/
├── qdrant_storage/                # Qdrant 컨테이너 볼륨 (벡터 데이터)
├── logs/
│   ├── ragent.log                 # 통합 로그 (모든 컴포넌트 전파)
│   ├── ragent_server.log          # RAGent 서버
│   ├── llm_server.log             # LLM 서버
│   ├── vectordb_server.log        # vectordb 컴포넌트
│   ├── ragent_server.console.log  # RAGent 서버 stdout/stderr 원본 캡처
│   └── llm_server.console.log     # LLM 서버 stdout/stderr 원본 캡처
└── failed_chunks.jsonl            # 영구 실패한 색인 항목 (진단용)
```

---

## Hook 설정 상세

`install.py`가 등록하는 hook 명령은 셸 비종속 형식이며 어댑터 선택자를 `--adapter` 인자로 박습니다.

```
"<python 실행파일 경로>" -m ragent --adapter <adapter_name>
```

예) Claude Code의 `~/.claude/settings.json`에 병합되는 구조:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "\"/path/to/.venv/bin/python\" -m ragent --adapter claude_code",
            "timeout": 5
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "\"/path/to/.venv/bin/python\" -m ragent --adapter claude_code",
            "timeout": 600
          }
        ]
      }
    ]
  }
}
```

**timeout** (Claude Code · Codex):

| 이벤트 | timeout | 의미 |
|---|---|---|
| `UserPromptSubmit` | 5초 | 서버에 `/search`를 호출하는 hook 호출 자체의 상한 |
| `Stop` | 600초 | `/save`는 큐에 적재만 하므로 보통 수초 내 끝나지만, 여유를 둔 상한 |

> Windsurf의 hooks.json은 스키마가 달라 entry에 `timeout` 필드 없이 `{"command": ...}`만 등록합니다.

**멱등성:** 재설치 시 `is_ragent_hook()`으로 기존 RAGent hook을 식별해 제거한 뒤 새로 추가합니다. 소스 모드 명령(`-m ragent`)과 frozen 모드 명령(`RAGent-hook`)을 모두 인식하므로 여러 번 실행해도 중복되지 않습니다.

---

## 모니터링과 에러 처리

RAGent는 **호스트 에이전트의 정상 동작을 방해하지 않는 것**을 최우선으로 합니다.

| 원칙 | 구현 |
|---|---|
| 항상 exit 0 | 클라이언트(`main.run`)는 모든 예외를 포착하고 `sys.exit(0)` |
| 로그만 남김 | stderr 출력 없이 `~/.ragent/logs/`에 기록 |
| 루프 방지 | Stop 핸들러에서 `stop_hook_active` 체크 |
| 서버 장애 격리 | 서버 호출 실패(`post_json`)는 로그 후 `None` 반환 → hook은 조용히 통과 |
| transient 재시도 | 색인 시 임베딩/Qdrant 호출은 3회 지수 백오프 재시도(`index_turn`) |
| 영구 실패 보존 | 재시도 실패 항목은 `failed_chunks.jsonl`에 append |
| 재랭킹 degrade | `Reranker`는 예외 시 batch_size=1 재시도 → 그래도 실패하면 균일 fallback 점수로 통과(전 청크 0점 방지) |
| enrichment/쿼리변환 격리 | LLM 호출 실패 시 prefix 생략 / 원문 쿼리로 폴백, 예외를 밖으로 던지지 않음 |

**수동 확인**

```bash
# 통합 로그
tail -f ~/.ragent/logs/ragent.log

# 서버 상태
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/stats
```

`/stats`는 큐 적재 수(`queue_pending`), 진행 중인 저장/검색, 마지막 완료 작업을 보고하며 launcher 콘솔 출력의 소스입니다.
