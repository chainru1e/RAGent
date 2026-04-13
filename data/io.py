import os
import torch
from qdrant_client import QdrantClient
from configs.config import QDRANT_DIR
from data.labels import LABEL2IDX


def fetch_from_qdrant(
    collection_name: str,
    label_key: str = "type",
    path: str = QDRANT_DIR,
    batch_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Qdrant에서 dense 벡터와 라벨을 추출

    Args:
        collection_name: Qdrant 컬렉션 이름
        label_key      : payload에서 라벨로 사용할 키 (default: "type")
        path           : Qdrant 로컬 경로
        batch_size     : 한 번에 가져올 포인트 수

    Returns:
        embeddings : (N, D) float32 Tensor
        labels     : (N,)   int64 Tensor
    """
    client = QdrantClient(path=path)

    all_vectors = []
    all_labels  = []
    offset      = None

    while True:
        response, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_vectors=["dense"],
            with_payload=[label_key],
        )

        if not response:
            break

        for point in response:
            label = point.payload.get(label_key)
            if label not in LABEL2IDX:
                print(f"[경고] 알 수 없는 라벨 '{label}' - 건너뜀")
                continue
            all_vectors.append(point.vector["dense"])
            all_labels.append(label)

        if next_offset is None:
            break
        offset = next_offset

    client.close()

    embeddings = torch.tensor(all_vectors, dtype=torch.float32)
    labels     = torch.tensor([LABEL2IDX[l] for l in all_labels], dtype=torch.long)

    print(f"[fetch] {len(embeddings)}개 벡터 | dim={embeddings.shape[1]}")

    return embeddings, labels


def save_to_pt(embeddings: torch.Tensor, labels: torch.Tensor, save_path: str):
    """
    벡터와 라벨을 pt 파일로 저장

    Args:
        embeddings: (N, D) Tensor
        labels    : (N,)   Tensor
        save_path : 저장 경로
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        "embeddings": embeddings,
        "labels"    : labels,
    }, save_path)

    print(f"[저장 완료] {save_path} | {len(embeddings)}개 벡터, dim={embeddings.shape[1]}")


def load_from_pt(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    pt 파일에서 벡터와 라벨 로드

    Returns:
        embeddings : (N, D) Tensor
        labels     : (N,)   Tensor
    """
    data = torch.load(path)

    print(f"[로드 완료] {path} | {len(data['embeddings'])}개 벡터, dim={data['embeddings'].shape[1]}")

    return data["embeddings"], data["labels"]