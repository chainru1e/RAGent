from ragent.models.intent import IntentCategory

class ChunkMetaData:
    def __init__(self,
                 chunk_id: str = None,
                 parent_id: str = None,
                 file_path: str = None,
                 type: IntentCategory = None,
                 context_prefix: str = None,
                 workspace_id: str = None,
                 source_kind: str = "conversation",
                 snapshot_id: str = None,
                 snapshot_version: int = None,
                 is_current: bool = None,
                 indexed_at: str = None,
                 content_hash: str = None,
                 language: str = None):
        self.chunk_id = chunk_id
        self.parent_id = parent_id
        self.file_path = file_path
        self.type = type
        self.context_prefix = context_prefix
        self.workspace_id = workspace_id
        self.source_kind = source_kind
        self.snapshot_id = snapshot_id
        self.snapshot_version = snapshot_version
        self.is_current = is_current
        self.indexed_at = indexed_at
        self.content_hash = content_hash
        self.language = language

class Chunk:
    def __init__(self,
            metadata: ChunkMetaData,
            payload: str,
            vector = None):
        self.metadata = metadata
        self.payload = payload
        self.vector = vector
