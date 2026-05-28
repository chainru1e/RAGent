from ragent.models.intent import IntentCategory

class ChunkMetaData:
    def __init__(self,
                 chunk_id: str = None,
                 parent_id: str = None,
                 file_path: str = None,
                 type: IntentCategory = None,
                 context_prefix: str = None,
                 func_name: str = None,       # 추가: AST로 추출한 함수/클래스 이름
                 is_latest: bool = True,       # 추가: 최신 버전 여부 (RAG 필터 기준)
                 version: int = 1):            # 추가: 버전 카운터
        self.chunk_id = chunk_id
        self.parent_id = parent_id
        self.file_path = file_path
        self.type = type
        self.context_prefix = context_prefix
        self.func_name = func_name
        self.is_latest = is_latest
        self.version = version

class Chunk:
    def __init__(self,
            metadata: ChunkMetaData,
            payload: str,
            vector = None):
        self.metadata = metadata
        self.payload = payload
        self.vector = vector
