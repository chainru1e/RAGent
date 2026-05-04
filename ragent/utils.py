import psutil
import subprocess

def get_system_ram_gb():
    """시스템의 전체 RAM 용량을 GB 단위로 반환"""
    return psutil.virtual_memory().total / (1024**3)

def get_nvidia_vram_gb():
    """NVIDIA GPU의 VRAM 용량을 GB 단위로 반환 (없으면 0)"""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], 
            encoding='utf-8',
            stderr=subprocess.DEVNULL
        )
        return int(result.strip().split('\n')[0]) / 1024
    except Exception:
        return 0