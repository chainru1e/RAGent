import argparse
from pathlib import Path
from data.io import fetch_from_qdrant, save_to_pt


def export_embeddings(collection_name: str, save_path: Path):
    """
    Qdrant에서 dense 벡터와 라벨을 추출해 pt 파일로 저장

    Args:
        collection_name: Qdrant 컬렉션 이름
        save_path      : pt 파일 저장 경로
    """
    print(f"[export] '{collection_name}' 에서 데이터 추출 중...")
    embs, labels = fetch_from_qdrant(collection_name)

    save_to_pt(embs, labels, save_path)
    print(f"[export] 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, required=True, help="Qdrant 컬렉션 이름")
    args = parser.parse_args()

    save_path = Path("data") / f"{args.collection}_embeddings.pt"
    export_embeddings(collection_name=args.collection, save_path=save_path)