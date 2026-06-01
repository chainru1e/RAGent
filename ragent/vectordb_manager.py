from __future__ import annotations

import logging
import subprocess
import time

import requests

from ragent.config import QDRANT_HOST, QDRANT_PORT, QDRANT_DIR
from ragent.logging_config import setup_logging

logger = logging.getLogger("ragent.vectordb")

CONTAINER_NAME = "ragent-qdrant"
HEALTH_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}/healthz"


class QdrantManager:
    def is_running(self) -> bool:
        """Qdrant 서버가 응답하는지 확인"""
        try:
            response = requests.get(HEALTH_URL, timeout=2)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _is_docker_available(self) -> bool:
        """Docker가 설치되어 있는지 확인"""
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _is_docker_running(self) -> bool:
        """Docker Desktop이 실행 중인지 확인"""
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True
        )
        return result.returncode == 0

    def _start_existing_container(self) -> bool:
        """기존 컨테이너 시작 시도. 성공 시 True, 실패 시 False 반환"""
        result = subprocess.run(
            ["docker", "start", CONTAINER_NAME],
            capture_output=True
        )
        return result.returncode == 0

    def _create_container(self) -> bool:
        """새 컨테이너 생성. 성공 시 True, 실패 시 False 반환"""
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", CONTAINER_NAME,
                "-p", f"{QDRANT_PORT}:6333",
                "-v", f"{QDRANT_DIR}:/qdrant/storage",
                "qdrant/qdrant"
            ],
            capture_output=True
        )
        return result.returncode == 0

    def _wait_until_ready(self, timeout: int = 300) -> bool:
        """서버가 뜰 때까지 대기"""
        for _ in range(timeout):
            if self.is_running():
                return True
            time.sleep(1)
        return False

    def start(self) -> None:
        """Qdrant 서버 시작. 이미 실행 중이면 아무것도 하지 않음."""
        if self.is_running():
            logger.info("Qdrant server is already running")
            return

        if not self._is_docker_available():
            raise RuntimeError(
                "Docker is not installed.\n"
                "Please install Docker to proceed."
            )

        if not self._is_docker_running():
            raise RuntimeError(
                "Docker Desktop is not running.\n"
                "Please start Docker Desktop and try again."
            )

        logger.info("Starting Qdrant server...")

        if not self._start_existing_container():
            logger.info("Container not found. Creating a new one.")
            if not self._create_container():
                raise RuntimeError(
                    "Failed to create Qdrant container.\n"
                    "Please check Docker logs for more details."
                )

        if not self._wait_until_ready():
            raise RuntimeError("Failed to start Qdrant server (timeout)")

        logger.info("Qdrant server started successfully")

    def stop(self) -> None:
        """Qdrant 컨테이너 중지"""
        subprocess.run(
            ["docker", "stop", CONTAINER_NAME],
            capture_output=True
        )
        logger.info("Qdrant server stopped successfully")


def main() -> None:
    setup_logging()
    qdrant = QdrantManager()
    qdrant.start()


if __name__ == "__main__":
    main()