from qdrant_client import QdrantClient
from ragent.config import QDRANT_DIR

client = QdrantClient(path=QDRANT_DIR)

# ====== 컬렉션 목록 ======
print("\n========== 컬렉션 목록 ==========")
collections = client.get_collections()
for collection in collections.collections:
    print(f"  - {collection.name}")

# ====== 각 컬렉션 내부 확인 ======
for collection in collections.collections:
    print(f"\n========== [{collection.name}] 상태 ==========")
    info = client.get_collection(collection.name)
    print(f"  저장된 포인트 수: {info.points_count}")
    print(f"  상태: {info.status}")

    print(f"\n---------- [{collection.name}] 포인트 ----------")
    points, _ = client.scroll(
        collection_name=collection.name,
        limit=100,
        with_payload=True,
        with_vectors=True
    )

    if not points:
        print("  (비어있음)")
        continue

    for point in points:
        print(f"\n  [ID: {point.id}]")
        print(f"    chunk_id  : {point.payload.get('chunk_id')}")
        print(f"    parent_id : {point.payload.get('parent_id')}")
        print(f"    file_path : {point.payload.get('file_path')}")
        print(f"    type      : {point.payload.get('type')}")
        print(f"    text      : {point.payload.get('text', '')[:80]}{'...' if len(point.payload.get('text', '')) > 80 else ''}")
        print(f"    text      : {point.payload.get('text', '')}")

    #     vectors = point.vector
    #     if vectors:
    #         dense = vectors.get("dense")
    #         sparse = vectors.get("sparse")
    #         if dense:
    #             print(f"    dense     : {dense[:5]} ... (총 {len(dense)} 차원)")
    #         if sparse:
    #             print(f"    sparse    : indices: {sparse.indices[:5]} ... (총 {len(sparse.indices)} 토큰)")

client.close()