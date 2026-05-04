from ragent.utils import *
from ragent.config import LLM_REPO_ID, LLM_FILENAME, LLM_SERVER_HOST, LLM_SERVER_PORT
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import sys
import multiprocessing
import uvicorn

class LLMServer:
    def __init__(self,  repo_id=LLM_REPO_ID, filename=LLM_FILENAME):
        try:
            self.model_path = hf_hub_download(
                repo_id=repo_id, 
                filename=filename
            )
        except (RepositoryNotFoundError, EntryNotFoundError):
            # 레포지토리가 없거나 파일명이 틀렸을 때
            print(f"\n[Fatal Error] Model not found!")
            print(f"Please verify that the repository '{repo_id}' contains the file '{filename}'.")
            sys.exit(1)
        except Exception as e:
            # 기타 예외 상황
            print(f"\n[Unknown Error] An unexpected error occurred during the download: {e}")
            sys.exit(1)
        
        # 하드웨어 스펙 스캔
        self.ram_gb = get_system_ram_gb()
        self.vram_gb = get_nvidia_vram_gb()
        self.cpu_cores = multiprocessing.cpu_count()

    def _calculate_optimal_settings(self):
        """현재 시스템의 하드웨어 스펙을 바탕으로 LLM 구동을 위한 최적의 파라미터를 계산한다.

        미리 스캔된 VRAM, RAM 용량과 CPU 코어 수를 기반으로, Qwen 3.5 9B 모델이
        안정적이고 빠르게 동작할 수 있도록 GPU 오프로딩 수준과 컨텍스트 길이를 동적으로 할당한다.

        Returns:
            dict: llama.cpp 엔진 구동에 필요한 설정값을 담은 딕셔너리.
                - n_gpu_layers (int): GPU 메모리에 올릴 모델의 레이어 수
                  (-1: 전체 오프로딩, 0: CPU 전용, 20: 부분 오프로딩)
                - n_ctx (int): 모델이 한 번에 처리할 수 있는 최대 컨텍스트(토큰) 길이
                - n_threads (int): 모델 추론에 할당할 최적의 CPU 스레드 수
        """
        optimal_threads = max(1, self.cpu_cores - 1)
        
        settings_dict = {
            "n_gpu_layers": 0,
            "n_ctx": 2048,
            "n_threads": optimal_threads
        }
        
        # Qwen 3.5 9B (약 5.56 GB 차지, 32 layers) 기준 하드웨어 분기
        if self.vram_gb >= 8:
            settings_dict["n_gpu_layers"] = -1
            settings_dict["n_ctx"] = 8192
            
        elif self.vram_gb >= 4:
            settings_dict["n_gpu_layers"] = 20
            settings_dict["n_ctx"] = 4096
            
        elif self.ram_gb >= 15:
            settings_dict["n_gpu_layers"] = 0
            settings_dict["n_ctx"] = 4096
            
        else:
            settings_dict["n_gpu_layers"] = 0
            settings_dict["n_ctx"] = 2048
            
        return settings_dict
    
    def start_server(self, host=LLM_SERVER_HOST, port=LLM_SERVER_PORT):
        optimal_params = self._calculate_optimal_settings()
        
        # llama-cpp-python 서버의 Settings 객체에 값 주입
        settings = Settings(
            model=self.model_path,
            n_gpu_layers=optimal_params["n_gpu_layers"],
            n_ctx=optimal_params["n_ctx"],
            n_threads=optimal_params["n_threads"],
            host=host,
            port=port
        )
        
        app = create_app(settings=settings)
        
        uvicorn.run(app, host=settings.host, port=settings.port)

if __name__ == "__main__":
    client = LLMServer()
    client.start_server()