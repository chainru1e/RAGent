import logging
import subprocess
import sys
import threading
import time
from typing import Any

import requests

from ragent.config import LLM_API_BASE_URL, RAGENT_SERVER_URL
from ragent.logging_config import setup_logging
from ragent.vectordb_manager import QdrantManager

logger = logging.getLogger("ragent")

POLL_INTERVAL_SECONDS = 2
HEARTBEAT_SECONDS = 30


def _wait_until_ready(proc: subprocess.Popen, url: str, name: str, timeout: int = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited unexpectedly during startup (code={proc.returncode})")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"{name} did not become ready within {timeout}s")


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _short(session_id: str | None) -> str:
    return (session_id or "?")[:8]


def _is_idle(data: dict[str, Any]) -> bool:
    return (
        data.get("queue_pending", 0) == 0
        and not data.get("currently_processing")
        and not data.get("currently_searching")
    )


def _emit_state_changes(prev: dict[str, Any], curr: dict[str, Any]) -> None:
    """Compare two /stats snapshots and print only the meaningful transitions."""
    prev_pending = prev.get("queue_pending", 0)
    curr_pending = curr.get("queue_pending", 0)

    prev_proc = prev.get("currently_processing")
    curr_proc = curr.get("currently_processing")
    prev_proc_sid = prev_proc.get("session_id") if prev_proc else None
    curr_proc_sid = curr_proc.get("session_id") if curr_proc else None

    prev_search = prev.get("currently_searching")
    curr_search = curr.get("currently_searching")
    prev_search_sid = prev_search.get("session_id") if prev_search else None
    curr_search_sid = curr_search.get("session_id") if curr_search else None

    prev_last_save = prev.get("last_completed")
    curr_last_save = curr.get("last_completed")
    prev_last_search = prev.get("last_search")
    curr_last_search = curr.get("last_search")

    if curr_search and curr_search_sid != prev_search_sid:
        print(f"[{_ts()}] Search started — session={_short(curr_search_sid)}")

    if curr_last_search and curr_last_search != prev_last_search:
        s = curr_last_search
        sid = _short(s.get("session_id"))
        dur = s.get("duration_s", 0)
        if s.get("status") == "ok":
            print(
                f"[{_ts()}] Search done — {s.get('result_count', 0)} chunks "
                f"({s.get('context_len', 0)} chars), {dur:.1f}s (session={sid})"
            )
        else:
            print(f"[{_ts()}] Search failed — {dur:.1f}s (session={sid})")

    if curr_pending > prev_pending:
        added = curr_pending - prev_pending
        suffix = f" (+{added})" if added > 1 else ""
        print(f"[{_ts()}] Save queued — {curr_pending} pending{suffix}")

    if curr_proc and curr_proc_sid != prev_proc_sid:
        print(
            f"[{_ts()}] Save started — {curr_pending} pending "
            f"(session={_short(curr_proc_sid)})"
        )

    if curr_last_save and curr_last_save != prev_last_save:
        s = curr_last_save
        sid = _short(s.get("session_id"))
        dur = s.get("duration_s", 0)
        chunks = s.get("chunks_indexed", 0)
        if s.get("status") == "ok":
            print(f"[{_ts()}] Save done — {chunks} chunks, {dur:.1f}s (session={sid})")
        else:
            print(f"[{_ts()}] Save failed — {dur:.1f}s (session={sid})")


def _emit_heartbeat(data: dict[str, Any]) -> None:
    """Single-line reminder so long-running work stays visible at HEARTBEAT_SECONDS intervals."""
    parts: list[str] = []
    search = data.get("currently_searching")
    if search:
        parts.append(f"search {search.get('elapsed_s', 0):.0f}s elapsed")
    proc = data.get("currently_processing")
    if proc:
        parts.append(f"save {proc.get('elapsed_s', 0):.0f}s elapsed")
    if parts:
        print(f"  · {', '.join(parts)} — still working")


def _sleep_with_stop(seconds: float, stop_event: threading.Event) -> None:
    steps = max(1, int(seconds * 2))
    for _ in range(steps):
        if stop_event.is_set():
            return
        time.sleep(seconds / steps)


def _poll_stats(url: str, stop_event: threading.Event) -> None:
    """Poll /stats periodically; print only when the underlying state actually changes.

    Stays silent while only the elapsed clock ticks forward. Long-running work is
    surfaced via a single heartbeat line every HEARTBEAT_SECONDS. Server hiccups
    are swallowed quietly until the next successful poll.
    """
    prev: dict[str, Any] | None = None
    last_heartbeat = 0.0

    while not stop_event.is_set():
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code != 200:
                _sleep_with_stop(POLL_INTERVAL_SECONDS, stop_event)
                continue
            data = resp.json()
        except (requests.exceptions.RequestException, ValueError):
            _sleep_with_stop(POLL_INTERVAL_SECONDS, stop_event)
            continue

        if prev is None:
            if _is_idle(data):
                print(f"[{_ts()}] Idle — no work in progress")
            else:
                _emit_heartbeat(data)
                last_heartbeat = time.time()
        else:
            _emit_state_changes(prev, data)

        now = time.time()
        active_elapsed = max(
            (data.get("currently_searching") or {}).get("elapsed_s", 0),
            (data.get("currently_processing") or {}).get("elapsed_s", 0),
        )
        if active_elapsed >= HEARTBEAT_SECONDS and now - last_heartbeat >= HEARTBEAT_SECONDS:
            _emit_heartbeat(data)
            last_heartbeat = now
        elif _is_idle(data):
            last_heartbeat = 0.0

        prev = data
        _sleep_with_stop(POLL_INTERVAL_SECONDS, stop_event)


def main() -> None:
    setup_logging()

    llm_proc = None
    server_proc = None
    stop_event = threading.Event()

    try:
        # 1. Qdrant Docker 컨테이너 — 준비될 때까지 블로킹
        print("[1/3] Starting Qdrant...")
        QdrantManager().start()
        print("[1/3] Qdrant ready")
        logger.info("Qdrant ready")

        # 2. LLM 서버 — 최대 5분 대기
        print("[2/3] Starting LLM server...")
        llm_proc = subprocess.Popen(
            [sys.executable, "-m", "ragent.llm_server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_until_ready(llm_proc, f"{LLM_API_BASE_URL}/models", "LLM server")
        print("[2/3] LLM server ready")
        logger.info("LLM server ready (pid=%d)", llm_proc.pid)

        # 3. RAGent 서버 — 최대 5분 대기
        print("[3/3] Starting RAGent server...")
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "ragent.server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_until_ready(server_proc, f"{RAGENT_SERVER_URL}/health", "RAGent server")
        print("[3/3] RAGent server ready")
        logger.info("RAGent server ready (pid=%d)", server_proc.pid)

        # 콘솔에는 사용자가 직관적으로 볼 수 있는 상태 변화만, 상세 로그는 파일로.
        threading.Thread(
            target=_poll_stats,
            args=(f"{RAGENT_SERVER_URL}/stats", stop_event),
            name="ragent-stats-poll",
            daemon=True,
        ).start()

        print("\nAll servers running.")
        print("Detailed logs: ~/.ragent/logs/ragent.log")
        print("Press Ctrl+C to stop.\n")

        while True:
            for proc, name in [(llm_proc, "LLM server"), (server_proc, "RAGent server")]:
                code = proc.poll() # 하위 프로세스 상태 확인
                if code is not None: # 프로세스 종료 여부 감지
                    logger.error("%s (pid=%d) exited unexpectedly with code %d", name, proc.pid, code)
                    print(f"\n[ERROR] {name} exited unexpectedly (code={code}). Shutting down.")
                    raise SystemExit(1)
            time.sleep(2)

    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        print(f"\n[ERROR] {e}")
        logger.error("Startup error: %s", e)
    finally:
        # 폴링 스레드 중단
        stop_event.set()

        # 1. RAGent 서버 종료
        if server_proc is not None:
            print("\nStopping RAGent server...")
            logger.info("Stopping RAGent server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

        # 2. LLM 서버 종료
        if llm_proc is not None:
            print("Stopping LLM server...")
            logger.info("Stopping LLM server...")
            llm_proc.terminate()
            try:
                llm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                llm_proc.kill()

        # 3. Qdrant Docker 컨테이너 종료
        print("Stopping Qdrant...")
        logger.info("Stopping Qdrant...")
        QdrantManager().stop()
        logger.info("All servers stopped")
        print("All servers stopped")


if __name__ == "__main__":
    main()
