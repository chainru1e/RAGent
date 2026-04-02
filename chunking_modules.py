# 턴 단위 청킹
# 코드 AST 기반 청킹
# 메타데이터 기반 계층적 청킹
# https://arxiv.org/html/2506.15655v1
# https://github.com/yilinjz/astchunk

from astchunk import ASTChunkBuilder
import os
import queue
import threading
import copy
import uuid

# +-----------------------------------+
# |          Observer Pattern         |
# +-------------+---------------------+
# |  Subject    |  TurnManager        |
# +-------------+---------------------+
# |  Observer   |  AsyncChunker       |
# +-------------+---------------------+

class TurnManager:
    """
    파싱된 메시지들을 수집하여 일련의 과정이 포함된 하나의 완전한 대화 턴을 관리하는 클래스.

    Observer pattern의 Subject 역할을 수행한다.
    단일 턴 내에서 발생하는 개별 메시지들을 버퍼에 누적하며, 메시지의 'stop_reason'이 
    'end_turn'에 도달해 해당 턴이 완전히 종료되면 등록된 Observer들에게 수집된 전체 턴 데이터를 전달한다.

    Attributes:
        buffer (list): 현재 진행 중인 턴의 메시지 딕셔너리들을 임시로 저장하는 리스트.
        observers (list): 턴 종료 알림을 받을 Observer 객체들의 리스트.
                          - 필수 구현: `on_turn_ended(turn_data)`: 턴 데이터 수신용
                          - 선택 구현: `wait_until_done()`: 비동기 Observer에 한해, 
                                       시스템 종료 시 백그라운드 작업이 끝날 때까지 대기하도록 하는 메서드
    """
    def __init__(self):
        self.buffer = []
        self.observers = []

    def attach(self, observer):
        self.observers.append(observer)

    def notify_turn_ended(self, turn_data):
        for observer in self.observers:
            observer.on_turn_ended(turn_data)

    def process_message(self, msg: dict):
        """
        message_parser에서 나온 대화를 버퍼에 저장하고 턴 종료를 감지한다
        Args:
            msg (dict): message_parser에서에서 파싱된 딕셔너리
        """
        self.buffer.append(msg)
        if msg.get("stop_reason") == "end_turn":
            self.notify_turn_ended(copy.copy(self.buffer))
            self.buffer.clear()
    
    def wait_all_observers(self):
        for observer in self.observers:
            if hasattr(observer, "wait_until_done"):
                observer.wait_until_done()

class ChunkMetaData:
    def __init__(self,
                 chunk_id: str = None,
                 parent_id: str = None,
                 file_path: str = None):
        self.chunk_id = chunk_id
        self.parent_id = parent_id
        self.file_path = file_path

    def to_dict(self):
        pass

class Chunk:
    def __init__(self,
            metadata: ChunkMetaData,
            payload: str,
            vector = None):
        self.metadata = metadata
        self.payload = payload
        self.vector = vector
    
    def to_dict(self):
        pass

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

class AsyncChunker:
    """
    Observer pattern을 따르며, 수집된 턴 데이터를 비동기적으로 청킹 처리하는 클래스.

    TurnManager(Subject)로부터 완성된 대화 턴 데이터를 전달받아 내부 작업 큐에 저장한 뒤,
    별도의 백그라운드 스레드에서 다음의 변환 파이프라인을 비동기로 실행한다:
    1. 턴 데이터에서 자연어 문맥과 코드 블록을 분리 추출
    2. 생성된/수정된 코드를 대상 언어의 AST 기반으로 세분화(Chunking)
    3. 분리된 정보들을 부모(문맥) - 자식(코드) 형태의 계층형 문서 목록으로 패키징
    4. 최종 생성된 문서들을 지정된 출력 큐(Output Queue)에 전달

    이러한 구조를 통해 메인 스레드는 파일 파싱이나 모델 API 호출과 같은 무거운 작업의 
    병목 없이 메시지 수집을 지속할 수 있다.

    - 내부적으로 tree-sitter 파서를 사용하므로, 필요한 파서 바이너리를 
    추가 설치하여 지원 언어를 유연하게 확장할 수 있다.

    Attributes:
        task_queue (queue.Queue): TurnManager로부터 전달받은 턴 데이터를 임시 보관하는 큐.
        output_queue (queue.Queue | None): 청킹이 완료된 최종 문서를 전달할 목적지 큐.
        worker_thread (threading.Thread): 큐에서 데이터를 꺼내 처리하는 데몬 스레드.
        configs (dict): ASTChunkBuilder 설정값(청크 크기, 메타데이터 템플릿 등)을 담은 딕셔너리.
        builders_cache (dict): 언어별로 생성된 ASTChunkBuilder 인스턴스를 재사용하기 위한 캐시.
    """
    def __init__(self, output_queue=None):
        self.task_queue = queue.Queue()
        self.output_queue = output_queue
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # ASTChunkBuilder 설정
        self.configs = {
            "max_chunk_size": 1000,        # 청크당 최대 문자 수 (공백 제외)
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

    def on_turn_ended(self, turn_data):
        self.task_queue.put(turn_data)
    
    def extract_turn_components(self, turn_data: list[dict]) -> tuple[Chunk, list[Chunk]]:
        """
        하나의 대화 턴에서 문맥 텍스트와 코드 블록을 분리 추출하고, 이를 `Chunk` 객체로 포장한다.

        이 함수는 turn_data에 포함된 메시지들을 순회하면서 다음 두 가지를 만든다.
        1. context_chunk: [text] 태그가 붙은 일반 대화 텍스트를 역할 정보와 함께 누적한 문맥(부모) 청크
        2. code_chunks: ASSISTANT의 [Write] 메시지에서 파일 경로와 코드 본문을 추출하여 조립한 코드(자식) 청크 리스트

        대화 턴 전체를 묶는 고유 UUID(id)를 생성하여 문맥과 코드를 하나로 묶는 데이터 계층 구조의 기반을 마련한다.

        처리 규칙:
        - 각 메시지의 role은 기본값 unknown으로 읽고 대문자로 정규화한다.
        - content가 [text]로 시작하면 태그를 제거해 context_text에 추가한다.
        - role이 ASSISTANT이고 content가 [Write]로 시작하면 파일 경로와 코드 본문을 파싱한다.
        - 파싱된 코드는 `ChunkMetadata`가 부여된 도립적인 `Chunk` 객체로 조립되어 리스트에 추가된다.
        - 루프 종료 후, context_text 또한 `ChunkMetadata`가 부여된 단일 `Chunk`객체로 조립된다.

        Args:
            turn_data (list[dict]): 한 턴에 속한 메시지 딕셔너리 목록.
                각 원소는 보통 role, content 키를 가진다.

        Returns:
           context_chunk,code_chunks (tuple[chunk,list[chunk]]):
                - context_chunk (Chunk): 누적된 문맥 문자열을 담고 있는 청크 객체
                - code_chunks (list[chunk]): 추출된 개별 코드 블록들을 담고 있는 청크 객체 리스트

        Notes:
            - 형식이 맞지 않는 메시지는 건너뛰도록 설계되어 있다.
        """
        context_text = ""
        code_chunks = []

        id = str(uuid.uuid4())

        for msg in turn_data:
            role = msg.get("role", "unknown").upper() # USER 또는 ASSISTANT
            content = msg.get("content", "").strip()

            if content.startswith("[text]"):
                pure_text = content.replace("[text]", "", 1).strip()
                context_text += f"[{role}] {pure_text}\n"
            elif role == "ASSISTANT" and content.startswith("[Write]"):
                # 줄바꿈 단위로 쪼개서 파일명과 코드를 분리
                lines = content.split("\n")
                
                # lines[0] : "[Write]"
                # lines[1] : 파일 경로
                # lines[2:]: 실제 코드 내용
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

    def split_code_by_ast(self, code_chunks: list[Chunk]) -> list[Chunk]:
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

    def _worker_loop(self):
        """
        비동기 청킹 작업을 수행하는 백그라운드 스레드의 메인 루프.

        작업 큐(task_queue)에서 턴 단위 대화 데이터(turn_data)를 지속적으로 가져와 다음 작업을 수행한다:
        1. 객체 초기 포장 (`extract_turn_components`): 
           원시 데이터를 분석하여 고유 식별자(parent_id)를 발급하고, 
           문맥과 코드를 각각 완벽한 `Chunk` 객체(부모-자식)로 생성한다.
        2. 의미론적 세분화 (`split_code_by_ast`): 
           코드 `Chunk` 객체들을 AST 기반으로 쪼개면서 기존 메타데이터를 안전하게 유전시킨다.
        3. 객체 병합 및 전달: 
           맥락 청크와 세분화된 코드 청크들을 하나의 리스트로 묶어, 
           다음 파이프라인을 담당할 output_queue에 `Chunk` 객체 리스트 상태로 전달한다.

        이 메서드는 데몬 스레드에서 무한히 실행되며, 프로그램 종료 또는 명시적인 중단 신호가 
        있을 때까지 대기 상태와 작업 상태를 반복한다. 처리 완료 시 `task_done()`을 호출한다.
        """
        while True:
            turn_data = self.task_queue.get()
            context_chunk, code_chunks = self.extract_turn_components(turn_data)
            refined_code_chunks = self.split_code_by_ast(code_chunks)
            final_chunks = [context_chunk] + refined_code_chunks
            if self.output_queue is not None:
                self.output_queue.put(final_chunks)
            self.task_queue.task_done()
    
    def wait_until_done(self):
        self.task_queue.join()