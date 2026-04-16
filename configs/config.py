from pathlib import Path
from adapters.supcon_adapter import SupConAdapter, supcon_infonce_loss
from adapters.sup_ot_adapter import SupOTAdapter, sup_ot_loss

QDRANT_DIR = Path.home() / ".ragent" / "qdrant_storage"

ADAPTER_COMPONENTS = {
    "supcon": {
        "model": SupConAdapter,
        "loss": supcon_infonce_loss
    },
    "sup_ot": {
        "model": SupOTAdapter,
        "loss": sup_ot_loss
    }
}

def get_adapter_components(adapter_type: str):
    if adapter_type not in ADAPTER_COMPONENTS:
        raise ValueError(f"[오류] 지원하지 않는 어댑터 타입입니다: {adapter_type}\n"
                         f"사용 가능한 타입: {list(ADAPTER_COMPONENTS.keys())}")
    
    components = ADAPTER_COMPONENTS[adapter_type]
    return components["model"], components["loss"]