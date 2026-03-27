# RAGent

Claude Code의 대화 내역을 자동으로 벡터 데이터베이스에 저장하고, 이후 유사한 과거 대화를 검색할 수 있게 하는 RAG(Retrieval-Augmented Generation) 기반 인덱서입니다. Claude Code hooks 시스템과 연동되어 사용자가 의식하지 않아도 모든 대화가 자동으로 수집·색인됩니다.

---

## 아키텍처

RAGent는 Claude Code의 3가지 hook 시점에 개입하여 대화 데이터를 수집합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Claude Code Session                         │
└─────────────────────────────────────────────────────────────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
 ┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
 │ UserPrompt   │         │    Stop      │         │   SessionEnd     │
 │ Submit       │         │              │         │                  │
 │ (timeout 5s) │         │ (timeout 600s│         │ (timeout 600s)   │
 └──────┬───────┘         └──────┬───────┘         └────────┬─────────┘
        │                        │                          │
        ▼                        ▼                          ▼
 ┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
 │ 프롬프트를   │         │ pending +    │         │ 전체 transcript  │
 │ pending/에   │         │ transcript → │         │ 파싱 → 모든 Q&A  │
 │ 임시 저장    │         │ Q&A 1쌍 색인 │         │ upsert + 세션    │
 │              │         │              │         │ 요약 생성·색인   │
 └──────┬───────┘         └──────┬───────┘         └────────┬─────────┘
        │                        │                          │
        ▼                        ▼                          ▼
 ~/.ragent/pending/       ~/.ragent/chroma_db/       ~/.ragent/chroma_db/
   {session}.json         ├─ qa_pairs (컬렉션)       ├─ qa_pairs (upsert)
                          └─ (1쌍 인덱싱)            └─ session_summaries
```

**데이터 흐름 요약:**
1. **UserPromptSubmit** — 사용자가 프롬프트를 제출하면, 응답이 오기 전에 프롬프트를 `~/.ragent/pending/`에 임시 저장합니다.
2. **Stop** — Claude가 응답을 완료하면, 임시 저장된 프롬프트와 transcript의 마지막 응답을 짝지어 ChromaDB에 색인합니다.
3. **SessionEnd** — 세션이 종료되면, 전체 transcript를 파싱하여 모든 Q&A 쌍을 upsert하고, 세션 요약을 생성·색인합니다.

---

## 파일 구조

```
RAGent/
├── pyproject.toml                     # 패키지 메타데이터, 의존성 정의
├── install.py                         # Claude Code hooks 자동 등록 스크립트
├── ragent/
│   ├── __init__.py                    # 패키지 마커
│   ├── __main__.py                    # python -m ragent 진입점
│   ├── config.py                      # 경로 상수 및 디렉토리 초기화
│   ├── main.py                        # stdin JSON 파싱 → 핸들러 디스패치
│   ├── pending.py                     # 프롬프트 임시 파일 atomic write/read/delete
│   ├── transcript.py                  # JSONL transcript 파싱 및 Turn 추출
│   ├── vectordb.py                    # ChromaDB 래퍼 (RAGentDB 클래스)
│   └── handlers/
│       ├── __init__.py                # 패키지 마커
│       ├── user_prompt_submit.py      # UserPromptSubmit 이벤트 핸들러
│       ├── stop.py                    # Stop 이벤트 핸들러
│       └── session_end.py             # SessionEnd 이벤트 핸들러
└── tests/
    ├── __init__.py
    ├── test_transcript.py             # transcript 파싱 테스트 (9개 케이스)
    └── fixtures/
        └── sample_transcript.jsonl    # 테스트용 JSONL 픽스처
```

---

## 설치 및 실행 방법

### 요구 사항

- Python 3.10 이상
- Claude Code CLI 설치 및 실행 가능한 환경

### 1. 의존성 설치

```bash
cd RAGent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

개발 의존성(pytest)까지 설치하려면:

```bash
pip install -e ".[dev]"
```

### 2. Claude Code hooks 등록

```bash
# venv가 활성화된 상태에서 실행
python install.py
```

이 명령은 `~/.claude/settings.json`에 RAGent hook 설정을 자동으로 병합합니다. 기존 설정이 있으면 유지하면서 RAGent 항목만 추가/갱신합니다.

### 3. 자동 동작 원리

설치 후에는 별도 실행이 필요 없습니다. Claude Code가 세션 중 hook 이벤트를 발생시키면, stdin으로 JSON 데이터를 전달하며 다음 명령을 호출합니다:

```bash
PYTHONPATH=/path/to/RAGent /path/to/RAGent/.venv/bin/python -m ragent
```

RAGent는 JSON에서 `hook_event_name` 필드를 읽어 적절한 핸들러로 라우팅합니다.

---

## 각 모듈 상세 설명

### `config.py` — 경로 및 상수

모든 파일 경로와 컬렉션 이름을 중앙에서 관리합니다.

| 상수 | 값 | 설명 |
|------|-----|------|
| `BASE_DIR` | `~/.ragent/` | 루트 디렉토리 |
| `PENDING_DIR` | `~/.ragent/pending/` | 프롬프트 임시 파일 저장소 |
| `CHROMA_DIR` | `~/.ragent/chroma_db/` | ChromaDB 영속 저장소 |
| `LOG_FILE` | `~/.ragent/ragent.log` | 디버그 로그 파일 |
| `COLLECTION_QA` | `"qa_pairs"` | Q&A 쌍 컬렉션 이름 |
| `COLLECTION_SUMMARIES` | `"session_summaries"` | 세션 요약 컬렉션 이름 |

`ensure_dirs()` 함수가 필요한 디렉토리를 자동 생성합니다.

### `pending.py` — 프롬프트 임시 저장

프롬프트를 `{session_id}.json` 파일로 저장합니다. **atomic write** 패턴(임시 파일 생성 → `os.rename`)을 사용하여 불완전한 파일이 남지 않도록 합니다.

- `save_pending_prompt(session_id, prompt, timestamp)` — 임시 파일에 쓴 후 rename
- `load_pending_prompt(session_id)` → `dict | None` — 파일이 없거나 손상 시 `None` 반환
- `delete_pending_prompt(session_id)` — 파일이 없어도 에러 없이 처리

### `transcript.py` — JSONL 파싱

Claude Code가 생성하는 JSONL transcript 파일을 파싱하여 `Turn` 객체 리스트로 변환합니다.

**파싱 규칙:**
- `isMeta: true`인 엔트리는 건너뜀
- `content`가 문자열이 아닌 엔트리(tool_result 등)는 건너뜀
- `<command-name>`, `<local-command-`로 시작하는 내용은 건너뜀
- assistant 응답에서 `type: "thinking"` 블록은 제외하고 `type: "text"` 블록만 추출
- 연속된 user → assistant 엔트리를 한 쌍(Turn)으로 묶음

**주요 함수:**
- `parse_transcript(path)` → `list[Turn]` — 전체 파싱
- `get_last_turn(path)` → `Turn | None` — 마지막 턴만 추출
- `get_session_summary_text(turns)` → `str` — 턴 목록을 요약 텍스트로 변환

### `vectordb.py` — ChromaDB 래퍼

`RAGentDB` 클래스가 2개의 ChromaDB 컬렉션을 관리합니다.

**컬렉션 구조:**

| 컬렉션 | 문서 형식 | ID | 메타데이터 |
|---------|-----------|-----|-----------|
| `qa_pairs` | `"Q: {프롬프트}\nA: {응답[:1000]}"` | `user_uuid` 또는 해시 | `timestamp`, `prompt_length`, `response_length` |
| `session_summaries` | 세션 요약 텍스트 | `session_id` | `session_id`, `turn_count`, `first_prompt` |

**주요 메서드:**
- `index_qa_pair(turn)` — Q&A 쌍 upsert
- `index_session_summary(session_id, summary, metadata)` — 세션 요약 upsert
- `search(query, collection, n_results, where)` — 벡터 유사도 검색

### `handlers/` — 이벤트 핸들러

#### `user_prompt_submit.py`
1. stdin JSON에서 `session_id`와 `prompt` 추출
2. 현재 UTC 타임스탬프 생성
3. `save_pending_prompt()` 호출하여 `~/.ragent/pending/{session_id}.json`에 저장
4. DB 접근 없이 빠르게 완료 (< 50ms)

#### `stop.py`
1. `stop_hook_active` 체크 — `True`이면 무한 루프 방지를 위해 즉시 반환
2. `load_pending_prompt()`로 임시 저장된 프롬프트 로드
3. `get_last_turn()`으로 transcript에서 마지막 응답 추출
4. pending 프롬프트와 응답을 짝지어 `Turn` 생성
5. `RAGentDB().index_qa_pair()` 호출하여 색인
6. pending 파일 삭제

#### `session_end.py`
1. `parse_transcript()`로 전체 transcript 파싱
2. 모든 Q&A 쌍을 `index_qa_pair()`로 upsert (Stop이 놓친 것까지 보완)
3. `get_session_summary_text()`로 세션 요약 생성
4. `index_session_summary()`로 요약 색인
5. pending 파일 정리

### `main.py` — 디스패처

stdin에서 JSON을 읽고, `hook_event_name` 필드를 기준으로 적절한 핸들러를 호출합니다.

```
stdin JSON → json.loads() → hook_event_name 분기
  ├── "UserPromptSubmit" → handlers/user_prompt_submit.handle()
  ├── "Stop"             → handlers/stop.handle()
  ├── "SessionEnd"       → handlers/session_end.handle()
  └── 기타               → 경고 로그
```

모든 예외를 최상위에서 포착하고 로그에 기록합니다. **어떤 상황에서도 exit code 0**을 반환합니다.

---

## 데이터 저장 위치

```
~/.ragent/
├── pending/                  # 프롬프트 임시 파일 (세션별 JSON)
│   └── {session_id}.json     #   → Stop/SessionEnd에서 소비 후 삭제
├── chroma_db/                # ChromaDB 영속 저장소
│   ├── qa_pairs/             #   → Q&A 쌍 벡터 인덱스
│   └── session_summaries/    #   → 세션 요약 벡터 인덱스
└── ragent.log                # 디버그/에러 로그 (append 모드)
```

---

## 검색 API 사용법

RAGent가 색인한 데이터를 Python 코드에서 직접 검색할 수 있습니다.

### Q&A 쌍 검색

```python
from ragent.vectordb import RAGentDB
from ragent.config import COLLECTION_QA

db = RAGentDB()
results = db.search("Python 가상환경 설정 방법", n_results=3)

for r in results:
    print(f"[거리: {r['distance']:.4f}]")
    print(r["document"])
    print(f"  타임스탬프: {r['metadata']['timestamp']}")
    print()
```

### 세션 요약 검색

```python
from ragent.config import COLLECTION_SUMMARIES

results = db.search(
    "데이터베이스 마이그레이션",
    collection=COLLECTION_SUMMARIES,
    n_results=5,
)

for r in results:
    print(f"세션: {r['metadata']['session_id']}")
    print(f"턴 수: {r['metadata']['turn_count']}")
    print(r["document"][:200])
    print()
```

### where 필터 사용

```python
# 특정 조건으로 필터링 (ChromaDB where 문법)
results = db.search(
    "에러 핸들링",
    where={"prompt_length": {"$gt": 100}},
    n_results=5,
)
```

---

## 테스트 실행

### 자동 테스트

```bash
pytest
```

또는 상세 출력:

```bash
pytest -v
```

9개의 테스트 케이스가 transcript 파싱, 필터링, 요약 생성을 검증합니다.

### 수동 검증

hook이 정상 동작하는지 확인하려면:

```bash
# 로그 확인
tail -f ~/.ragent/ragent.log

# ChromaDB에 데이터가 색인되었는지 확인
python -c "
from ragent.vectordb import RAGentDB
db = RAGentDB()
print('Q&A pairs:', db._qa.count())
print('Summaries:', db._summaries.count())
"
```

---

## 에러 처리 정책

RAGent는 **Claude Code의 정상 동작을 절대 방해하지 않는 것**을 최우선으로 합니다.

| 원칙 | 구현 방식 |
|------|-----------|
| **항상 exit 0** | `main.py`에서 모든 예외를 포착한 후 `sys.exit(0)` 호출 |
| **로그만 남김** | stderr 출력 없이 `~/.ragent/ragent.log`에 기록 |
| **입력 검증** | 필수 필드 누락 시 경고 로그 후 조용히 반환 |
| **파일 안전** | pending 파일 로드 실패 시 `None` 반환, 삭제 실패 시 무시 |
| **atomic write** | 임시 파일 → rename 패턴으로 불완전 파일 방지 |
| **루프 방지** | Stop 핸들러에서 `stop_hook_active` 체크 |

**로그 위치:** `~/.ragent/ragent.log`

**fallback 전략:**
- pending 파일이 없으면 transcript에서 직접 마지막 턴을 추출
- transcript 파일이 없으면 빈 리스트 반환 후 조기 종료
- JSON 파싱 실패 시 해당 라인을 건너뛰고 계속 진행

---

## Hook 설정 상세

`install.py`를 실행하면 `~/.claude/settings.json`에 다음 구조가 병합됩니다:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "PYTHONPATH=/path/to/RAGent /path/to/RAGent/.venv/bin/python -m ragent",
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
            "command": "PYTHONPATH=/path/to/RAGent /path/to/RAGent/.venv/bin/python -m ragent",
            "timeout": 600
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "PYTHONPATH=/path/to/RAGent /path/to/RAGent/.venv/bin/python -m ragent",
            "timeout": 600
          }
        ]
      }
    ]
  }
}
```

**timeout 설정 의미:**

| 이벤트 | timeout | 이유 |
|--------|---------|------|
| `UserPromptSubmit` | 5초 | 파일 쓰기만 하므로 빠르게 완료되어야 함. 응답 시작을 지연시키지 않기 위해 짧게 설정 |
| `Stop` | 600초 | ChromaDB 초기화 및 색인 작업 포함. 최초 실행 시 모델 다운로드 등으로 시간이 걸릴 수 있음 |
| `SessionEnd` | 600초 | 전체 transcript 파싱 + 모든 Q&A upsert + 요약 생성·색인. 긴 세션일수록 시간 필요 |

**중복 방지:** `install.py`는 기존에 등록된 RAGent hook을 감지(`"ragent" in command`)하여 제거한 후 새로 추가합니다. 따라서 여러 번 실행해도 안전합니다.
