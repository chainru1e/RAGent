import psutil
import subprocess
import sys


def get_system_ram_gb() -> float:
    """시스템의 전체 RAM 용량을 GB 단위로 반환"""
    return psutil.virtual_memory().total / (1024 ** 3)


def _get_vram_gb_windows() -> float:
    """Windows 환경에서 GPU VRAM을 레지스트리에서 감지한다.

    WMI의 AdapterRAM 필드는 32비트라 4GB 이상의 VRAM을 올바르게 반환하지 못하는
    버그가 있다. 레지스트리의 HardwareInformation.qwMemorySize 키는 64비트로
    실제 VRAM 용량을 저장하므로 이를 사용한다.
    """
    import winreg  # Windows 전용 표준 라이브러리

    GPU_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    best = 0

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, GPU_CLASS) as cls_key:
            i = 0
            while True:
                try:
                    sub_name = winreg.EnumKey(cls_key, i)
                    i += 1
                    # "0000", "0001" 같은 어댑터 서브키만 처리
                    if not sub_name.isdigit():
                        continue
                    sub_path = f"{GPU_CLASS}\\{sub_name}"
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, sub_path) as sub_key:
                        try:
                            value, _ = winreg.QueryValueEx(sub_key, "HardwareInformation.qwMemorySize")
                            best = max(best, int(value))
                        except FileNotFoundError:
                            pass
                except OSError:
                    break  # 서브키 열거 완료
    except Exception:
        return 0.0

    return best / (1024 ** 3)  # bytes → GB


def _get_vram_gb_linux_nvidia() -> float:
    """Linux/macOS 환경에서 NVIDIA VRAM을 nvidia-smi로 감지한다."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        return int(result.strip().split("\n")[0]) / 1024  # MiB → GB
    except Exception:
        return 0.0


def _get_vram_gb_linux_amd() -> float:
    """Linux 환경에서 AMD VRAM을 ROCm 툴로 감지한다 (rocm-smi → rocminfo)."""
    try:
        result = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        for line in result.strip().splitlines():
            if "Total" in line or "vram" in line.lower():
                for part in line.split(","):
                    part = part.strip()
                    if part.isdigit():
                        return int(part) / (1024 ** 2)  # bytes → GB
    except Exception:
        pass

    try:
        result = subprocess.check_output(
            ["rocminfo"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        for line in result.splitlines():
            if "Size:" in line and "kB" in line:
                kb = int(line.split("Size:")[1].replace("kB", "").strip())
                return kb / (1024 ** 2)  # kB → GB
    except Exception:
        pass

    return 0.0


def get_vram_gb() -> float:
    """감지된 GPU의 VRAM 용량을 GB 단위로 반환 (없으면 0).

    OS별로 다른 감지 방법을 사용한다:
    - Windows: 레지스트리 HardwareInformation.qwMemorySize (NVIDIA/AMD 모두 커버)
    - Linux:   nvidia-smi → rocm-smi/rocminfo 순으로 시도

    Apple Silicon은 Unified Memory라 별도 감지하지 않는다.
    is_apple_silicon()을 사용할 것.
    """
    if sys.platform == "win32":
        return _get_vram_gb_windows()

    # Linux
    vram = _get_vram_gb_linux_nvidia()
    if vram > 0:
        return vram
    return _get_vram_gb_linux_amd()


def is_apple_silicon() -> bool:
    """Apple Silicon(M-series) 여부를 반환한다.

    llama.cpp는 Metal API를 명시적으로 호출해야 GPU를 사용한다.
    OS가 자동으로 GPU로 라우팅하지 않으므로, Apple Silicon 감지 후
    n_gpu_layers = -1을 설정해 Metal 가속을 활성화해야 한다.
    """
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.check_output(
            ["sysctl", "-n", "hw.optional.arm64"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        return result.strip() == "1"
    except Exception:
        return False