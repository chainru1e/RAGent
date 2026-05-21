import logging
import subprocess
import sys
import time

from ragent.logging_config import setup_logging
from ragent.vectordb_manager import QdrantManager

logger = logging.getLogger("ragent")


def main() -> None:
    setup_logging()

    # 1. Qdrant Docker 컨테이너 — 준비될 때까지 블로킹
    print("[1/3] Starting Qdrant...")
    QdrantManager().start()
    print("[1/3] Qdrant ready")
    logger.info("Qdrant ready")

    # 2. LLM 서버 — 모델 로딩이 오래 걸리므로 먼저 띄움
    print("[2/3] Starting LLM server...")
    llm_proc = subprocess.Popen(
        [sys.executable, "-m", "ragent.llm_server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("LLM server started (pid=%d)", llm_proc.pid)

    # 3. RAGent 서버
    print("[3/3] Starting RAGent server...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "ragent.server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("RAGent server started (pid=%d)", server_proc.pid)

    print("All servers running. Press Ctrl+C to stop.\n")

    procs = [llm_proc, server_proc]

    try:
        while True:
            for proc in procs:
                code = proc.poll() # 하위 프로세스 상태 확인
                if code is not None: # 프로세스 종료 여부 감지
                    name = "LLM server" if proc is llm_proc else "RAGent server"
                    logger.error("%s (pid=%d) exited unexpectedly with code %d", name, proc.pid, code)
                    print(f"\n[ERROR] {name} exited unexpectedly (code={code}). Shutting down.")
                    raise SystemExit(1)
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # 1. RAGent 서버 종료
        print("\nStopping RAGent server...")
        logger.info("Stopping RAGent server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()

        # 2. LLM 서버 종료
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
