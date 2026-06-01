"""Transcript 파서 모듈.

각 어댑터(Claude Code, Codex, ...) 의 transcript 한 줄 스키마를
NormalizedMessage 로 번역하는 파서들을 한 파일에 모은 모듈이다. 역순으로
가장 최근 1턴을 뽑는 알고리즘은 모든 파서가 공유하는 Template Method 로
BaseParser 에 둔다.

향후 GeminiCliParser 등 새로운 어댑터용 파서가 추가될 경우 동일한 BaseParser
를 상속하여 본 파일에 함께 정의한다.

prefix tag 컨벤션 ('[text]', '[<tool_name>]', '[tool_result]', '[thinking]',
'[Image Attached]') 은 다운스트림 chunker 와의 implicit 인터페이스이므로
모든 서브클래스가 동일하게 사용한다.
"""

import json
import logging
import os
from abc import ABC, abstractmethod

from ragent.models.parsed_message import Block, NormalizedMessage

logger = logging.getLogger("ragent")


class BaseParser(ABC):
    """Transcript 파서 추상 베이스.

    각 어댑터(Claude Code, Codex, ...) 는 자신의 transcript 한 줄 스키마를
    NormalizedMessage 로 번역하는 서브클래스를 제공한다. 역순 1턴 추출
    알고리즘은 모든 파서가 공유하는 Template Method 로 이 베이스에 둔다.
    """

    def __init__(self, transcript_path: str):
        self.transcript_path = transcript_path

    @abstractmethod
    def parse_transcript_line(self, json_line: dict) -> NormalizedMessage | None:
        """이 어댑터의 transcript 한 줄(json 파싱된 dict) 을 정규화 메시지로 변환.

        형식이 안 맞거나 RAGent 가 의미 있는 메시지로 해석하지 않는 줄(예: 시스템
        이벤트, 빈 본문) 은 None 을 반환한다.
        """

    def _process_line(self, raw: bytearray) -> NormalizedMessage | None:
        """역순으로 누적된 buffer 를 디코딩·파싱해 NormalizedMessage 로 반환한다.

        빈 줄이거나 JSON 파싱에 실패하면 None 을 반환한다.
        """
        line = raw[::-1].decode('utf-8')
        if not line.strip():
            return None
        try:
            data = json.loads(line)
            return self.parse_transcript_line(data)
        except json.JSONDecodeError:
            return None

    def parse_last_turn(self) -> list[NormalizedMessage]:
        """Transcript 를 역순으로 읽어 가장 최근의 1턴을 추출한다.

        파일의 끝에서부터 바이트 단위로 거꾸로 읽어 올라가며 한 줄씩 파싱하고,
        사용자(user)의 실제 텍스트 입력('[text]') 을 턴의 시작점으로 간주하여
        탐색을 종료한다.
        """
        turn: list[NormalizedMessage] = []

        if not os.path.exists(self.transcript_path) or os.path.getsize(self.transcript_path) == 0:
            return turn

        with open(self.transcript_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            position = f.tell()
            buffer = bytearray()

            while position >= 0:
                f.seek(position)
                char = f.read(1)

                if char == b'\n' and buffer:
                    parsed_message = self._process_line(buffer)
                    buffer.clear()
                    if parsed_message:
                        turn.append(parsed_message)
                        if parsed_message.role == 'user' and parsed_message.content.startswith('[text]'):
                            turn.reverse()
                            return turn
                elif char != b'\n':
                    buffer.extend(char)

                position -= 1

            # 루프 종료 후 잔여 버퍼 처리 (첫번째 줄 처리)
            if buffer:
                parsed_message = self._process_line(buffer)
                if parsed_message:
                    turn.append(parsed_message)

        turn.reverse()
        return turn

    def build_file_snapshots(self) -> dict[str, str]:
        """transcript 전체에서 `[Write]` 본문을 모아 `file_path → 최신 본문` 맵을 만든다.

        파일 단위 Contextual Retrieval 의 경로 ②(세션 전체 latest Write 스냅샷)
        용. parse_last_turn 의 동일 turn Write(경로 ①) 가 잡지 못하는, 이전
        turn 에 쓰였거나 `[Edit]` 로만 등장한 파일의 문서 출처를 보강한다.

        전 라인을 시간순(파일 위→아래) 으로 `parse_transcript_line` 에 통과시켜
        role==assistant 이고 content 가 `[Write]` 로 시작하는 메시지에서
        lines[1]=path, "\\n".join(lines[2:])=body 를 모은다. 같은 파일이 여러 번
        쓰였으면 나중(아래쪽) Write 가 덮어쓴다. `[Edit]`/`MultiEdit` 은 파일 전체
        본문이 아니므로 제외한다.

        실패 격리: 파일 없음/빈 파일/깨진 라인은 건너뛰고, 어떤 입력에도 예외를
        밖으로 던지지 않는다(부분 결과 또는 빈 dict 반환).
        """
        snapshots: dict[str, str] = {}

        if not os.path.exists(self.transcript_path) or os.path.getsize(self.transcript_path) == 0:
            return snapshots

        try:
            with open(self.transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    try:
                        msg = self.parse_transcript_line(data)
                    except Exception:
                        # 파서는 throw 하지 않도록 설계되었으나, 스냅샷 수집이
                        # 인덱싱 흐름을 깨지 않도록 방어적으로 한 번 더 감싼다.
                        continue
                    if msg is None or msg.role != "assistant":
                        continue
                    content = msg.content
                    if not content.startswith("[Write]"):
                        continue
                    lines = content.split("\n")
                    if len(lines) < 2:
                        continue
                    path = lines[1].strip()
                    body = "\n".join(lines[2:]).strip()
                    if path and body:
                        snapshots[path] = body
        except OSError:
            logger.exception(
                "parser: failed to read transcript for file snapshots (%s)",
                self.transcript_path,
            )

        return snapshots


class ClaudeCodeParser(BaseParser):
    """Claude Code transcript JSONL 파서.

    형식: 각 줄은 대략 다음과 같은 JSON 객체.
        {"message": {"role": ..., "content": ... or [...]}, "timestamp": ...}
    'message' 키가 없는 줄(시스템 이벤트) 은 무시한다.

    출력 표현(전환기 이중 표현):
    - blocks: assistant 메시지의 content 배열 블록을 구조 보존한 Block 리스트.
      text/thinking → type + text, tool_use → name + 원시 input dict 그대로.
      tool_use 의 input 은 가공·삭제 없이 그대로 들고만 다닌다 — new_string 추출 /
      old_string 폐기 같은 의미 판단은 파서에 없다. 청커의 resolve_code_action
      단일 정의가 소유한다. 청커는 이 리스트를 순회하므로 한 메시지에
      text + tool_use 가 섞이거나 tool_use 가 여러 개여도 누락·오염 없이 처리된다.
    - content: 위 블록들을 prefix tag 와 함께 평탄화한 문자열. tool_use 는 종류를
      구분하지 않는 generic dump("[tool_name]\\n" + input.values()) 뿐이라 파서에
      edit 지식이 0 이다. turn 해시(결정론적 UUID5) 와 blocks 미이관 어댑터의
      폴백 경로용으로 유지한다. serialize_turn 은 더 이상 이 문자열의 코드 추출에
      의존하지 않고 blocks 를 직접 읽는다.
    - 비정상 블록(dict 아님 등) 은 skip 한다. 파서 전체는 어떤 입력에서도 throw
      하지 않는다.
    """

    def parse_transcript_line(self, json_line: dict) -> NormalizedMessage | None:
        # 시스템 이벤트 무시
        if 'message' not in json_line:
            return None

        msg_data = json_line['message']
        role = msg_data.get('role')
        content = msg_data.get('content')
        timestamp = json_line.get('timestamp')

        text_content = ""
        blocks: list[Block] = []

        # User 메시지 처리
        if role == 'user':
            if isinstance(content, str):
                text_content += "[text]\n"
                text_content += content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get('type')
                        if block_type == 'text':
                            text_content += "[text]\n"
                            text_content += block.get('text', '') + "\n"
                        elif block_type == 'tool_result':
                            text_content += "[tool_result]\n"
                            tool_content = block.get('content', '')
                            if isinstance(tool_content, str):
                                text_content += f"{tool_content}\n"
                            elif isinstance(tool_content, list):
                                for item in tool_content:
                                    text_content += f"{item}\n"
                        elif block_type == 'image':
                            text_content += "[Image Attached]\n"

        # Assistant 메시지 처리
        elif role == 'assistant':
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type')
                    if block_type == 'text':
                        text = block.get('text', '')
                        blocks.append(Block(type='text', text=text))
                        text_content += "[text]\n"
                        text_content += text + "\n"
                    elif block_type == 'thinking':
                        thinking = block.get('thinking', '')
                        blocks.append(Block(type='thinking', text=thinking))
                        text_content += "[thinking]\n"
                        text_content += thinking
                    elif block_type == 'tool_use':
                        tool_name = block.get('name')
                        tool_input = block.get('input', {})
                        # 원시 input dict 를 가공 없이 그대로 보존한다. 어느 키가
                        # 코드인지(edit 의미) 의 판단은 파서가 하지 않는다 — 청커의
                        # resolve_code_action 단일 정의가 소유한다.
                        blocks.append(Block(
                            type='tool_use',
                            name=tool_name,
                            input=tool_input if isinstance(tool_input, dict) else None,
                        ))
                        # content 는 generic dump 뿐. tool 종류를 구분하지 않으므로
                        # 파서에 edit 지식이 0 이다. (해시·폴백 호환용 평탄 표현)
                        text_content += f"[{tool_name}]\n"
                        if isinstance(tool_input, dict):
                            for val in tool_input.values():
                                text_content += f"{val}\n"

        if text_content.strip():
            return NormalizedMessage(
                role=role,
                content=text_content.strip(),
                blocks=blocks,
                timestamp=timestamp,
            )
        return None


class CodexParser(BaseParser):
    """Codex CLI rollout JSONL 파서.

    형식: 각 줄은 RolloutLine envelope.
        {"type": "<variant>", "payload": {...}, "timestamp": "..."}

    RolloutItem variants (출처: openai/codex codex-rs/protocol/src/models.rs):
    - session_meta   : 세션 메타데이터, 메시지 아님 → 무시
    - response_item  : OpenAI Responses API item — 사용자/어시스턴트 본문, tool_use,
                       tool_result, reasoning 등이 모두 이 타입의 inner type 으로
                       구분되어 저장된다. RAGent 가 의미 있게 다루는 줄.
    - compacted      : Codex 내부 압축 산출물 → 무시
    - turn_context   : 시스템 컨텍스트 메타 → 무시
    - event_msg      : 스트리밍 UI 이벤트 (UserMessage/AgentMessage delta 등). 본
                       파서는 response_item 만 표준 소스로 사용해 중복 인덱싱을
                       피한다 → 무시.

    response_item 의 inner type (ResponseItem enum):
    - message            : {role, content: [ContentItem]}.
                           ContentItem variant 는 input_text/input_image/output_text.
    - reasoning          : 어시스턴트 내부 reasoning. ClaudeCodeParser 의 thinking
                           에 대응 → "[thinking]" 으로 정규화.
    - function_call      : 어시스턴트 tool_use. ClaudeCodeParser 의 tool_use 와
                           동일한 컨벤션 "[<tool_name>]\\n<arguments>" 로 정규화.
    - function_call_output: tool 결과. role 은 user 로 부여 (Claude Code 가 user
                           메시지의 tool_result block 으로 표현하는 것과 일관) 하고
                           "[tool_result]" prefix 를 단다.
    - 그 외 (web_search_call, image_generation_call, local_shell_call, ...) :
      본 phase 에서 정규화 미정의 → None.

    chunker 호환성: ClaudeCodeParser 와 동일한 prefix tag 컨벤션
    ('[text]', '[<tool_name>]', '[tool_result]', '[thinking]') 을 그대로 사용한다.
    chunker 의 [Write] 리터럴 매칭은 Codex 에서는 발화하지 않는다 (Codex 의 파일
    쓰기는 apply_patch / shell 로 일어나고 Write 라는 이름의 tool 이 없음).
    context_text 인덱싱은 Claude Code 와 동일하게 동작.
    """

    def parse_transcript_line(self, json_line: dict) -> NormalizedMessage | None:
        if json_line.get("type") != "response_item":
            return None

        payload = json_line.get("payload") or {}
        if not isinstance(payload, dict):
            return None

        item_type = payload.get("type")
        timestamp = json_line.get("timestamp")

        if item_type == "message":
            role = payload.get("role")
            if role not in ("user", "assistant"):
                return None
            text_content = self._render_message_content(payload.get("content"))
            return self._wrap(role, text_content, timestamp)

        if item_type == "function_call":
            tool_name = payload.get("name", "")
            arguments = payload.get("arguments", "")
            text_content = f"[{tool_name}]\n"
            # arguments 는 JSON-encoded 문자열. 디코딩되면 ClaudeCodeParser 의
            # tool_input.values() 루프와 같은 모양으로 풀어 넣고, 디코딩 실패 시
            # 원시 문자열을 그대로 둔다.
            parsed_args = self._try_parse_json(arguments)
            if isinstance(parsed_args, dict):
                for val in parsed_args.values():
                    text_content += f"{val}\n"
            else:
                if arguments:
                    text_content += f"{arguments}\n"
            return self._wrap("assistant", text_content, timestamp)

        if item_type == "function_call_output":
            output = payload.get("output", "")
            text_content = "[tool_result]\n"
            if isinstance(output, str):
                text_content += f"{output}\n"
            elif isinstance(output, list):
                for item in output:
                    text_content += f"{item}\n"
            elif isinstance(output, dict):
                # 일부 변종: {"content": "..."} 또는 구조화된 결과. 안전하게 평탄화.
                inner = output.get("content", output)
                text_content += f"{inner}\n"
            return self._wrap("user", text_content, timestamp)

        if item_type == "reasoning":
            summary = payload.get("summary")
            content = payload.get("content")
            text_content = "[thinking]\n"
            text_content += self._flatten_text(summary)
            text_content += self._flatten_text(content)
            return self._wrap("assistant", text_content, timestamp)

        return None

    @staticmethod
    def _render_message_content(content) -> str:
        text_content = ""
        if isinstance(content, str):
            text_content += "[text]\n"
            text_content += content
            return text_content
        if not isinstance(content, list):
            return text_content
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in ("input_text", "output_text"):
                text_content += "[text]\n"
                text_content += block.get("text", "") + "\n"
            elif block_type == "input_image":
                text_content += "[Image Attached]\n"
        return text_content

    @staticmethod
    def _try_parse_json(value):
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _flatten_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value + "\n" if value else ""
        if isinstance(value, list):
            out = ""
            for item in value:
                if isinstance(item, dict):
                    out += item.get("text", "") + "\n"
                else:
                    out += f"{item}\n"
            return out
        return f"{value}\n"

    @staticmethod
    def _wrap(role: str, text_content: str, timestamp) -> NormalizedMessage | None:
        if not text_content.strip():
            return None
        return NormalizedMessage(
            role=role,
            content=text_content.strip(),
            timestamp=timestamp,
        )


class WindsurfParser(BaseParser):
    """Windsurf (Cognition Cascade) transcript JSONL 파서.

    형식: 각 줄은 {"type": "...", "status": "...", <type-specific>: {...}} envelope.
    위치: ~/.windsurf/transcripts/{trajectory_id}.jsonl 또는 hook payload 의
    post_cascade_response_with_transcript.tool_info.transcript_path 가 절대 경로
    제공.

    공식 docs (https://docs.windsurf.com/windsurf/cascade/hooks) 가 공개한 step
    type 은 3개뿐이며, "the exact structure of each step may change in future
    versions" 라는 명시적 경고가 함께 붙어 있다. 따라서 본 파서는 공개된 3개만
    정규화하고 그 외 9개 hook 이벤트 (read_code/run_command/mcp_tool_use 등)
    에 대응하는 미공개 type 은 모두 None 반환 — 추측 매핑 금지.

    공개된 type 매핑:
    - user_input.user_response          : 사용자 본문   → role=user,     "[text]\\n<text>"
    - planner_response.response         : 어시스턴트 본문 → role=assistant, "[text]\\n<text>"
    - code_action.{path, new_content}   : 파일 작성       → role=assistant, "[code_action]\\n<path>\\n<new_content>"

    chunker 호환성: code_action 의 prefix tag 는 "[code_action]" 으로 부여한다.
    Claude Code 의 "[Write]" 리터럴과 비호환이므로 chunker 의 [Write] 매칭은
    발화하지 않음 — 코드 청크 인덱싱은 미작동 (Codex 의 apply_patch 와 동일
    함정). 추측으로 [Write] 마킹 강행하지 않음.

    parse_last_turn 은 BaseParser 의 Template Method 를 그대로 사용한다 —
    user '[text]' 메시지를 turn 시작점으로 보는 동일 규칙. parse_full_transcript
    는 커밋 8806cff 에서 BaseParser 에서 제거됐으므로 본 파서도 호출 경로 없음.
    """

    def parse_transcript_line(self, json_line: dict) -> NormalizedMessage | None:
        line_type = json_line.get("type")
        timestamp = json_line.get("timestamp")

        if line_type == "user_input":
            payload = json_line.get("user_input") or {}
            if not isinstance(payload, dict):
                return None
            text = payload.get("user_response", "")
            if not text:
                return None
            return NormalizedMessage(
                role="user",
                content=f"[text]\n{text}",
                timestamp=timestamp,
            )

        if line_type == "planner_response":
            payload = json_line.get("planner_response") or {}
            if not isinstance(payload, dict):
                return None
            text = payload.get("response", "")
            if not text:
                return None
            return NormalizedMessage(
                role="assistant",
                content=f"[text]\n{text}",
                timestamp=timestamp,
            )

        if line_type == "code_action":
            payload = json_line.get("code_action") or {}
            if not isinstance(payload, dict):
                return None
            path = payload.get("path", "")
            new_content = payload.get("new_content", "")
            return NormalizedMessage(
                role="assistant",
                content=f"[code_action]\n{path}\n{new_content}",
                timestamp=timestamp,
            )

        return None
