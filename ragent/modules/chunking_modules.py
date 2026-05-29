from astchunk import ASTChunkBuilder
import os
import copy
import uuid
import hashlib

from ragent.models.chunk import Chunk, ChunkMetaData
from ragent.models.parsed_message import Block, NormalizedMessage
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

    Attributes:
        configs (dict): ASTChunkBuilder 설정값(청크 크기, 메타데이터 템플릿 등)을 담은 딕셔너리.
        builders_cache (dict): 언어별로 생성된 ASTChunkBuilder 인스턴스를 재사용하기 위한 캐시.
    """

    UUID_NAMESPACE = uuid.NAMESPACE_OID

    def __init__(self):
        # ASTChunkBuilder 설정
        self.configs = {
            "max_chunk_size": MAX_CHUNK_SIZE,        # 청크당 최대 문자 수 (공백 제외)
            "metadata_template": "default" # 메타데이터 템플릿 형식
        }
        self.builders_cache = {}

    def _get_language_from_filename(self, file_path: str) -> str | None:
        """
        파일 경로에서 확장자를 추출하여 ASTChunkBuilder용 언어 이름을 반환합니다.
        
        Args:
            file_path (str): 확장자를 검사할 파일의 경로.

        Returns:
            str|None: 파서가 지원하는 언어 이름 문자열.
                      지원하지 않는 확장자이거나 확장자가 없는 경우 None을 반환한다.

        Notes:
            - 현재 지원 언어: Python (.py), Java (.java), C# (.cs), TypeScript (.ts), JavaScript (.js)
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # astchunk 지원 언어 매핑
        ext_to_lang = {
            ".py": "python",
            ".java": "java",
            ".cs": "csharp",
            ".ts": "typescript",
            ".js": "typescript"  # 통상적으로 js도 ts 파서로 처리 가능
        }
        return ext_to_lang.get(ext) # 지원하지 않는 확장자면 None 반환

    def _get_or_create_builder(self, language: str) -> ASTChunkBuilder:
        """
        주어진 언어에 대한 ASTChunkBuilder 인스턴스를 반환한다. (싱글톤 패턴 활용)

        언어별로 구문 분석기(Parser)를 초기화하는 비용을 줄이기 위해, 이미 생성된 
        빌더가 있다면 builders_cache에서 꺼내어 재사용한다. 
        캐시에 없다면 기본 설정을 복사하여 해당 언어의 빌더를 새로 생성하고 캐시에 저장한다.

        Args:
            language (str): 빌더를 생성할 언어 이름 (예: "python", "java").

        Returns:
            ASTChunkBuilder: 해당 언어를 파싱할 수 있도록 구성된 청크 빌더 인스턴스.
        """
        if language not in self.builders_cache:
            # 해당 언어의 빌더가 처음 요청된 경우 새로 생성
            configs = self.configs.copy()
            configs["language"] = language
            self.builders_cache[language] = ASTChunkBuilder(**configs)
            
        return self.builders_cache[language]
    
    def _extract_turn_components(self, turn_data: list[NormalizedMessage]) -> tuple[Chunk, list[Chunk]]:
        """
        하나의 대화 턴에서 문맥 텍스트와 코드 블록을 분리 추출하고, 이를 `Chunk` 객체로 포장한다.

        이 함수는 turn_data에 포함된 메시지들을 순회하면서 다음 두 가지를 만든다.
        1. context_chunk: text/thinking 블록의 본문을 역할 정보와 함께 누적한 문맥(부모) 청크
        2. code_chunks: ASSISTANT의 Write / Edit / MultiEdit tool_use 블록에서 파일 경로와
           코드 본문을 추출하여 조립한 코드(자식) 청크 리스트. tool_use 블록마다
           독립적인 코드 청크 1개를 만들므로, 한 메시지에 편집이 N개면 청크도 N개다.

        의미 판단(어느 input 키가 코드인지) 은 파서가 아니라 여기서 수행한다.
        - Write     → path=input["file_path"], code=input["content"]
        - Edit      → path=input["file_path"], code=input["new_string"] (old_string/replace_all 무시)
        - MultiEdit → path=input["file_path"], code="\\n".join(edits 의 new_string)
        - 그 외 tool_use(Read/Bash 등) 는 코드 청크를 만들지 않는다.

        turn_data에 포함된 메시지 전체 내용을 해시(Hash)하여 결정론적인 고유 UUID5(id)를 생성하고,
        이를 통해 문맥과 코드를 하나로 묶는 데이터 계층 구조의 기반을 마련한다.
        (동일한 대화 턴이 입력되면 항상 같은 ID를 보장하여 데이터 중복을 방지한다.)

        처리 규칙:
        - 각 메시지의 role은 대문자로 정규화한다.
        - blocks 가 채워진 메시지는 블록을 순회한다. text/thinking 블록은 context_text 에
          누적하고, ASSISTANT 의 tool_use 블록은 코드 청크 후보로 넘긴다.
        - blocks 가 빈 메시지(아직 blocks 로 이관되지 않은 어댑터: Codex/Windsurf) 는
          기존 content prefix([text]/[Write]/[Edit]) 경로로 폴백한다.
        - 파싱된 코드는 `ChunkMetadata`가 부여된 독립적인 `Chunk` 객체로 조립되어 리스트에 추가된다.
        - 필수 input 키가 없거나 코드가 빈 경우 해당 블록만 skip 한다 (throw 금지).
        - 루프 종료 후, context_text 또한 `ChunkMetadata`가 부여된 단일 `Chunk`객체로 조립된다.

        Args:
            turn_data (list[NormalizedMessage]): 한 턴에 속한 정규화 메시지 객체 목록.
                각 원소는 role, content 속성을 가진다.

        Returns:
           context_chunk,code_chunks (tuple[chunk,list[chunk]]):
                - context_chunk (Chunk): 누적된 문맥 문자열을 담고 있는 청크 객체
                - code_chunks (list[chunk]): 추출된 개별 코드 블록들을 담고 있는 청크 객체 리스트

        Notes:
            - 형식이 맞지 않는 메시지는 건너뛰도록 설계되어 있다.
        """
        context_text = ""
        code_chunks = []

        combined_content = "".join([f"{msg.role}:{msg.content}" for msg in turn_data])
        content_hash = hashlib.sha256(combined_content.encode('utf-8')).hexdigest()
        id = str(uuid.uuid5(self.UUID_NAMESPACE, content_hash))

        for msg in turn_data:
            role = msg.role.upper() # USER 또는 ASSISTANT

            if msg.blocks:
                # 구조 보존된 블록 경로: text/thinking 은 문맥으로, tool_use 는
                # 코드/비코드로 분기한다. 블록마다 독립 처리하므로 한 메시지에
                # text + tool_use 가 섞이거나 편집이 여러 개여도 누락·오염이 없다.
                for block in msg.blocks:
                    if block.type in ("text", "thinking"):
                        pure_text = (block.text or "").strip()
                        if pure_text:
                            context_text += f"[{role}] {pure_text}\n"
                    elif block.type == "tool_use" and role == "ASSISTANT":
                        code_chunk = self._build_code_chunk(block, id)
                        if code_chunk is not None:
                            code_chunks.append(code_chunk)
            else:
                # 폴백: 아직 blocks 로 이관되지 않은 어댑터(Codex/Windsurf) 의
                # 평탄 content prefix 경로. 이관 완료 시 제거 대상.
                content = msg.content.strip()
                if content.startswith("[text]"):
                    pure_text = content.replace("[text]", "", 1).strip()
                    context_text += f"[{role}] {pure_text}\n"
                elif role == "ASSISTANT" and (
                    content.startswith("[Write]") or content.startswith("[Edit]")
                ):
                    # 줄바꿈 단위로 쪼개서 파일명과 코드를 분리
                    # lines[0] : "[Write]" 또는 "[Edit]"
                    # lines[1] : 파일 경로
                    # lines[2:]: 실제 코드 내용 (Edit/MultiEdit 의 경우 new_string 조각들)
                    lines = content.split("\n")
                    if len(lines) >= 2:
                        file_path = lines[1].strip()
                        code_content = "\n".join(lines[2:]).strip()
                        if code_content:
                            code_metadata = ChunkMetaData(
                                parent_id=id,
                                file_path=file_path
                            )
                            code_chunks.append(Chunk(code_metadata, code_content))

        context_metadata = ChunkMetaData(chunk_id=id)
        context_chunk = Chunk(context_metadata, context_text)
        return context_chunk, code_chunks

    def _build_code_chunk(self, block: Block, parent_id: str) -> Chunk | None:
        """tool_use 블록 1개에서 코드 청크 1개를 조립한다 (코드 추출 의미 판단).

        Write / Edit / MultiEdit 만 코드 청크를 만든다. 어느 input 키가 인덱싱
        대상 코드인지의 판단을 여기서 수행한다. 필수 키가 없거나 코드가 비면
        그 블록만 skip(None 반환) 하고 절대 throw 하지 않는다.

        - Write     : file_path, content
        - Edit      : file_path, new_string  (old_string / replace_all 은 무시)
        - MultiEdit : file_path, edits[*].new_string 을 줄바꿈으로 결합
        """
        tool_input = block.input
        if not isinstance(tool_input, dict):
            return None

        name = block.name
        if name == "Write":
            if "file_path" not in tool_input or "content" not in tool_input:
                return None
            file_path = tool_input["file_path"]
            code = tool_input["content"]
        elif name == "Edit":
            if "file_path" not in tool_input or "new_string" not in tool_input:
                return None
            file_path = tool_input["file_path"]
            code = tool_input["new_string"]
        elif name == "MultiEdit":
            edits = tool_input.get("edits")
            if "file_path" not in tool_input or not isinstance(edits, list):
                return None
            pieces = [
                str(e["new_string"]) for e in edits
                if isinstance(e, dict) and "new_string" in e
            ]
            if not pieces:
                return None
            file_path = tool_input["file_path"]
            code = "\n".join(pieces)
        else:
            # Read / Bash 등 비코드 tool_use 는 코드 청크를 만들지 않는다.
            return None

        file_path = str(file_path).strip()
        code = str(code).strip()
        if not file_path or not code:
            return None

        code_metadata = ChunkMetaData(parent_id=parent_id, file_path=file_path)
        return Chunk(code_metadata, code)

    def _split_code_by_ast(self, code_chunks: list[Chunk]) -> list[Chunk]:
        """
        코드 `Chunk` 객체 리스트를 받아 AST 파서를 이용해 심화 청킹을 수행한다.

        입력된 각 `Chunk` 객체의 metadata.file_path 확장자를 검사하여 지원하는 언어인 경우,
        해당 언어의 ASTChunkBuilder를 사용하여 코드를 함수, 클래스 등의 의미 단위로 쪼갠다.
        지원하지 않는 언어거나 확장자를 알 수 없는 경우 `Chunk` 객체를 그대로 유지한다.

        Args:
            code_blocks (list[Chunk]): 원본 코드 블록 리스트. 

        Returns:
            list[Chunk]: AST 기반으로 세분화된 코드 `Chunk` 객체 리스트.
                하나의 원본 `Chunk` 객체가 의미 단위의 여러 `Chunk` 객체로 쪼개질 수 있으며, 
                쪼개진 모든 객체는 기존 객체의 메타데이터를 보존한 채로 반환된다.
        """
        refined_chunks = []
        for original_chunk in code_chunks:
            file_path = original_chunk.metadata.file_path
            raw_code = original_chunk.payload

            lang = self._get_language_from_filename(file_path)

            if lang:
                builder = self._get_or_create_builder(lang)
                ast_chunks = builder.chunkify(raw_code)
                for ast_chunk_data in ast_chunks:
                    new_metadata = copy.copy(original_chunk.metadata)
                    new_chunk = Chunk(
                        metadata=new_metadata,
                        payload=ast_chunk_data["content"]
                    )
                    refined_chunks.append(new_chunk)
            else:
                refined_chunks.append(original_chunk)
        
        return refined_chunks

    def process_turn(self, turn_data: list[NormalizedMessage]) -> list[Chunk]:
        """
        단일 대화 턴 데이터를 청킹하여 Chunk 리스트를 반환한다.

        파이프라인:
        1. 객체 초기 포장 (`extract_turn_components`): 
           원시 데이터를 분석하여 고유 식별자(parent_id)를 발급하고, 
           문맥과 코드를 각각 완벽한 `Chunk` 객체(부모-자식)로 생성한다.
        2. 의미론적 세분화 (`split_code_by_ast`): 
           코드 `Chunk` 객체들을 기존 메타데이터를 유지한 채 AST 기반으로 쪼갠다.
        3. 객체 병합 및 전달: 
           단일 맥락 청크 1개와 세분화된 코드 청크들을 하나의 리스트로 묶어 즉시 반환한다.

        Args:
            turn_data (list[NormalizedMessage]): 파서를 통해 추출된 1턴 분량의 정규화 메시지 객체 리스트.

        Returns:
            list[Chunk]: 처리가 완료된 최종 Chunk 객체들의 리스트. 
                         (항상 인덱스 0에는 맥락 청크가 위치하며, 이어서 코드 청크들이 배치된다.)
        """
        context_chunk, code_chunks = self._extract_turn_components(turn_data)
        refined_code_chunks = self._split_code_by_ast(code_chunks)
        # 최종적으로 쪼개진 결과에 chunk_id 부여
        parent_id = context_chunk.metadata.chunk_id
        for i, chunk in enumerate(refined_code_chunks):
            chunk.metadata.chunk_id = str(uuid.uuid5(self.UUID_NAMESPACE, f"{parent_id}_code_{i}"))
        return [context_chunk] + refined_code_chunks