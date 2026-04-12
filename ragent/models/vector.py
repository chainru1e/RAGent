import numpy as np
from qdrant_client.models import SparseVector

class HybridVector:
    def __init__(self, dense: np.ndarray, sparse: SparseVector):
        self.dense = dense
        self.sparse = sparse