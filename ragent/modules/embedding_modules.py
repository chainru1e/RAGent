from ragent.config import SPARSE_EMBEDDING_MODEL, DENSE_EMBEDDING_REPO_ID, DENSE_EMBEDDING_FILENAME, DENSE_EMBEDDING_N_CTX, DENSE_EMBEDDING_N_SEQ_MAX, DENSE_EMBEDDING_N_CTX_SEQ
from qdrant_client.models import SparseVector
from ragent.models.vector import HybridVector
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from fastembed import SparseTextEmbedding
from llama_cpp import Llama
import llama_cpp.llama_cpp as _llama_cpp_lib
import numpy as np
import math
import multiprocessing

class HybridEmbedding:
    def __init__(self):
        try:
            self.dense_model_path = hf_hub_download(
                repo_id=DENSE_EMBEDDING_REPO_ID,
                filename=DENSE_EMBEDDING_FILENAME,
            )
        except (RepositoryNotFoundError, EntryNotFoundError) as e:
            raise RuntimeError(f"Embedding model not found: repo={DENSE_EMBEDDING_REPO_ID} file={DENSE_EMBEDDING_FILENAME}") from e
        except Exception as e:
            raise RuntimeError("Unexpected error during embedding model download") from e
        
        _orig_max_parallel = _llama_cpp_lib.llama_max_parallel_sequences
        _llama_cpp_lib.llama_max_parallel_sequences = lambda: DENSE_EMBEDDING_N_SEQ_MAX
        try:
            self.dense_model = Llama(
                model_path=self.dense_model_path,
                embedding=True,
                pooling_type=-1,
                n_gpu_layers=0,
                n_ctx=DENSE_EMBEDDING_N_CTX,
                n_batch=DENSE_EMBEDDING_N_CTX_SEQ,
                n_threads=max(1, math.floor(multiprocessing.cpu_count() * 0.25)),
                verbose=False
            )
        finally:
            _llama_cpp_lib.llama_max_parallel_sequences = _orig_max_parallel

        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL)

    def embed(self, text: str) -> HybridVector:
        dense_output = self.dense_model.create_embedding(text)
        dense_vec = dense_output["data"][0]["embedding"]
        
        sparse_output = list(self.sparse_model.embed([text]))[0]

        return HybridVector(
            dense=np.array(dense_vec, dtype=np.float32),
            sparse=SparseVector(
                indices=sparse_output.indices.tolist(),
                values=sparse_output.values.tolist()
            )
        )

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[HybridVector]:
        sparse_outputs = list(self.sparse_model.embed(texts, batch_size=batch_size))
        dense_outputs = self.dense_model.create_embedding(texts)
        return [
            HybridVector(
                dense=np.array(dense_outputs["data"][i]["embedding"], dtype=np.float32),
                sparse=SparseVector(
                    indices=sparse_outputs[i].indices.tolist(),
                    values=sparse_outputs[i].values.tolist()
                )
            )
            for i in range(len(texts))
        ]