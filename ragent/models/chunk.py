from ragent.models.intent import IntentCategory
from enum import Enum

class ChunkMetaData:
    def __init__(self,
                 chunk_id: str = None,
                 parent_id: str = None,
                 file_path: str = None,
                 type: IntentCategory = None):
        self.chunk_id = chunk_id
        self.parent_id = parent_id
        self.file_path = file_path
        self.type = type

class Chunk:
    def __init__(self,
            metadata: ChunkMetaData,
            payload: str,
            vector = None):
        self.metadata = metadata
        self.payload = payload
        self.vector = vector