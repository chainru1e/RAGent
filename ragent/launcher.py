import logging
import subprocess
import sys
import time

import requests

from ragent.config import LLM_API_BASE_URL, RAGENT_SERVER_URL
from ragent.logging_config import setup_logging
from ragent.vectordb_manager import QdrantManager

logger = logging.getLogger("ragent")


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


def main() -> None:
    setup_logging()

    llm_proc = None
    server_proc = None

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

        print("\nAll servers running. Press Ctrl+C to stop.")

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
