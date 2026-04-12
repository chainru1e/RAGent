from ragent.config import EMBEDDING_MODEL
from FlagEmbedding import BGEM3FlagModel
from qdrant_client.models import SparseVector
from ragent.models.vector import HybridVector
import numpy as np

class HybridEmbedding:
    def __init__(self):
        self.model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)

    def embed(self, text: str) -> HybridVector:
        output = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=True
        )
        sparse = output["lexical_weights"][0]  # {token_id: weight} dict

        return HybridVector(
            dense=np.array(output["dense_vecs"][0], dtype=np.float32),
            sparse=SparseVector(
                indices=list(sparse.keys()),
                values=list(sparse.values())
            )
        )

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[HybridVector]:
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            output = self.model.encode(
                batch,
                return_dense=True,
                return_sparse=True
            )
            for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"]):
                vectors.append(
                    HybridVector(
                        dense=np.array(dense, dtype=np.float32),
                        sparse=SparseVector(
                            indices=list(sparse.keys()),
                            values=list(sparse.values())
                        )
                    )
                )
        return vectors