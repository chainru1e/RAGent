from astchunk import ASTChunkBuilder
import os
import copy
import uuid
import hashlib

from ragent.models.chunk import Chunk, ChunkMetaData
from ragent.models.parsed_message import NormalizedMessage
from ragent.config import MAX_CHUNK_SIZE

# =====================================================================
# [ ASTChunkBuilder 지원 언어 추가 가이드 (예: C++) ]
# =====================================================================
# 1. 패키지 설치
#    $ pip install tree-sitter-cpp
#
# 2. astchunk_builder.py 파일 열기
#    (경로: C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\astchunk\astchunk_builder.py)
#
# 3. 모듈 임포트 추가 (파일 상단)
#    import tree_sitter_cpp as tscpp
#
# 4. 파서 등록 (ASTChunkBuilder 클래스 __init__ 메서드 내)
#    elif self.language == "cpp":
#        self.parser = ts.Parser(ts.Language(tscpp.language()))
# =====================================================================


class Chunker:
    """
    단일 대화 턴 데이터를 청킹 처리하는 클래스.

    - 내부적으로 tree-sitter 파서를 사용하므로, 필요한 파서 바이너리를
    추가 설치하여 지원 언어를 유연하게 확장할 수 있다.
    - Edit 처리 시 VectorDB를 주입받아 기존 청크 버전 관리를 수행한다.

    Attributes:
        configs (dict): ASTChunkBuilder 설정값을 담은 딕셔너리.
        builders_cache (dict): 언어별로 생성된 ASTChunkBuilder 인스턴스 캐시.
        storage: QdrantStorage 인스턴스 (Edit 처리 시 버전 조회/Soft Delete용).
                 None이면 Edit도 Write처럼 처리한다.
    """

    UUID_NAMESPACE = uuid.NAMESPACE_OID

    def __init__(self, storage=None):
        self.configs = {
            "max_chunk_size": MAX_CHUNK_SIZE,
            "metadata_template": "default"
        }
        self.builders_cache = {}
        self.storage = storage  # 추가: QdrantStorage 의존성 주입

    def _get_language_from_filename(self, file_path: str) -> str | None:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        ext_to_lang = {
            ".py": "python",
            ".java": "java",
            ".cs": "csharp",
            ".ts": "typescript",
            ".js": "typescript"
        }
        return ext_to_lang.get(ext)

    def _get_or_create_builder(self, language: str) -> ASTChunkBuilder:
        if language not in self.builders_cache:
            configs = self.configs.copy()
            configs["language"] = language
            self.builders_cache[language] = ASTChunkBuilder(**configs)
        return self.builders_cache[language]

    def _extract_func_name_from_ast(self, code: str, language: str) -> str | None:
        """
        코드 조각에서 AST를 이용해 최상위 함수/클래스 이름을 추출한다.

        ASTChunkBuilder로 청킹한 결과의 첫 번째 청크 이름을 사용한다.
        추출에 실패하면 None을 반환한다.

        Args:
            code (str): 함수/클래스 단위의 코드 조각.
            language (str): 파싱할 언어 이름.

        Returns:
            str | None: 추출된 함수/클래스 이름. 실패 시 None.

        Notes:
            - astchunk의 반환 딕셔너리에 "name" 키가 있으면 이를 사용한다.
            - 없을 경우 코드 첫 줄에서 def/class 키워드로 이름을 파싱한다.
        """
        try:
            builder = self._get_or_create_builder(language)
            chunks = builder.chunkify(code)
            if chunks:
                # astchunk가 "name" 키를 제공하는 경우
                if "name" in chunks[0]:
                    return chunks[0]["name"]

            # fallback: 첫 줄에서 직접 파싱 (def login(...) → "login")
            first_line = code.strip().splitlines()[0]
            for keyword in ("def ", "class ", "function ", "func "):
                if keyword in first_line:
                    after = first_line.split(keyword, 1)[1]
                    name = after.split("(")[0].split(":")[0].strip()
                    return name if name else None
        except Exception:
            pass
        return None

    def _extract_turn_components(self, turn_data: list[NormalizedMessage]) -> tuple[Chunk, list[Chunk], list[bool]]:
        """
        하나의 대화 턴에서 문맥 텍스트와 코드 블록을 분리 추출하고, 이를 Chunk 객체로 포장한다.

        기존 동작에서 [Edit] 태그 여부를 is_edit 플래그로 구분하여 함께 반환한다.

        Returns:
            context_chunk, code_chunks, is_edit_flags (tuple):
                - context_chunk (Chunk): 누적된 문맥 청크
                - code_chunks (list[Chunk]): 코드 청크 리스트
                - is_edit_flags (list[bool]): 각 코드 청크가 Edit인지 여부
        """
        context_text = ""
        code_chunks = []
        is_edit_flags = []  # 추가: 각 코드 청크의 Edit 여부

        combined_content = "".join([f"{msg.role}:{msg.content}" for msg in turn_data])
        content_hash = hashlib.sha256(combined_content.encode('utf-8')).hexdigest()
        id = str(uuid.uuid5(self.UUID_NAMESPACE, content_hash))

        for msg in turn_data:
            role = msg.role.upper()
            content = msg.content.strip()

            if content.startswith("[text]"):
                pure_text = content.replace("[text]", "", 1).strip()
                context_text += f"[{role}] {pure_text}\n"

            elif role == "ASSISTANT" and content.startswith("[Write]"):
                lines = content.split("\n")
                if len(lines) >= 2:
                    file_path = lines[1].strip()
                    code_content = "\n".join(lines[2:]).strip()
                    if code_content:
                        code_metadata = ChunkMetaData(parent_id=id, file_path=file_path)
                        code_chunks.append(Chunk(code_metadata, code_content))
                        is_edit_flags.append(False)  # Write

            elif role == "ASSISTANT" and content.startswith("[Edit]"):
                lines = content.split("\n")
                if len(lines) >= 2:
                    file_path = lines[1].strip()
                    code_content = "\n".join(lines[2:]).strip()
                    if code_content:
                        code_metadata = ChunkMetaData(parent_id=id, file_path=file_path)
                        code_chunks.append(Chunk(code_metadata, code_content))
                        is_edit_flags.append(True)   # Edit

        context_metadata = ChunkMetaData(chunk_id=id)
        context_chunk = Chunk(context_metadata, context_text)
        return context_chunk, code_chunks, is_edit_flags

    def _split_code_by_ast(
        self,
        code_chunks: list[Chunk],
        is_edit_flags: list[bool]
    ) -> list[Chunk]:
        """
        코드 Chunk 리스트를 AST 파서로 함수/클래스 단위로 세분화하고,
        func_name / is_latest / version 메타데이터를 결정한다.

        [Write] 청크:
            - func_name: AST에서 추출
            - is_latest: True
            - version: 1

        [Edit] 청크:
            - func_name: AST에서 추출
            - storage가 있으면 기존 버전 조회 후 Soft Delete
            - is_latest: True
            - version: 기존 버전 + 1 (없으면 1)

        Args:
            code_chunks (list[Chunk]): 원본 코드 청크 리스트.
            is_edit_flags (list[bool]): 각 청크의 Edit 여부.

        Returns:
            list[Chunk]: 메타데이터가 완성된 세분화 코드 청크 리스트.
        """
        refined_chunks = []

        for original_chunk, is_edit in zip(code_chunks, is_edit_flags):
            file_path = original_chunk.metadata.file_path
            raw_code = original_chunk.payload
            lang = self._get_language_from_filename(file_path)

            if lang:
                builder = self._get_or_create_builder(lang)
                ast_chunks = builder.chunkify(raw_code)

                for ast_chunk_data in ast_chunks:
                    new_metadata = copy.copy(original_chunk.metadata)
                    code_piece = ast_chunk_data["content"]

                    # func_name 추출
                    func_name = None
                    if "name" in ast_chunk_data:
                        # astchunk가 직접 제공하는 경우
                        func_name = ast_chunk_data["name"]
                    else:
                        # fallback: 코드에서 직접 파싱
                        func_name = self._extract_func_name_from_ast(code_piece, lang)

                    new_metadata.func_name = func_name

                    if is_edit and self.storage and func_name:
                        # Edit 처리: 기존 버전 조회 → Soft Delete → 새 버전 세팅
                        current_version = self.storage.get_latest_version(file_path, func_name)

                        if current_version is not None:
                            # 기존 청크를 is_latest: False로 변경
                            self.storage.mark_outdated(file_path, func_name)
                            new_metadata.version = current_version + 1
                        else:
                            # DB에 없는 함수 → Write처럼 처리
                            new_metadata.version = 1

                        new_metadata.is_latest = True

                    else:
                        # Write 처리 또는 storage 없음
                        new_metadata.is_latest = True
                        new_metadata.version = 1

                    new_chunk = Chunk(metadata=new_metadata, payload=code_piece)
                    refined_chunks.append(new_chunk)

            else:
                # 지원하지 않는 언어: 메타데이터 기본값 유지
                original_chunk.metadata.is_latest = True
                original_chunk.metadata.version = 1
                refined_chunks.append(original_chunk)

        return refined_chunks

    def process_turn(self, turn_data: list[NormalizedMessage]) -> list[Chunk]:
        """
        단일 대화 턴 데이터를 청킹하여 Chunk 리스트를 반환한다.

        파이프라인:
        1. 객체 초기 포장: 문맥/코드 분리, Edit 플래그 발급
        2. 의미론적 세분화: AST 청킹 + func_name/버전 메타데이터 결정
        3. chunk_id 부여 후 병합 반환

        Args:
            turn_data (list[NormalizedMessage]): 1턴 분량의 정규화 메시지 리스트.

        Returns:
            list[Chunk]: 처리 완료된 Chunk 리스트.
                         인덱스 0: 맥락 청크, 이후: 코드 청크.
        """
        context_chunk, code_chunks, is_edit_flags = self._extract_turn_components(turn_data)
        refined_code_chunks = self._split_code_by_ast(code_chunks, is_edit_flags)

        parent_id = context_chunk.metadata.chunk_id
        for i, chunk in enumerate(refined_code_chunks):
            chunk.metadata.chunk_id = str(uuid.uuid5(self.UUID_NAMESPACE, f"{parent_id}_code_{i}"))

        return [context_chunk] + refined_code_chunks
