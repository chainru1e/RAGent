"""Transcript 파서 모듈.

각 어댑터(Claude Code, Codex, ...) 의 transcript 한 줄 스키마를
NormalizedMessage 로 번역하는 파서들을 한 파일에 모은 모듈이다. 메시지 스트림
위에서 도는 알고리즘(역순 1턴 추출, 전체 transcript turn 그룹핑) 은 모든
파서가 공유하는 Template Method 로 BaseParser 에 둔다.

향후 GeminiCliParser 등 새로운 어댑터용 파서가 추가될 경우 동일한 BaseParser
를 상속하여 본 파일에 함께 정의한다.

prefix tag 컨벤션 ('[text]', '[<tool_name>]', '[tool_result]', '[thinking]',
'[Image Attached]') 은 다운스트림 chunker 와의 implicit 인터페이스이므로
모든 서브클래스가 동일하게 사용한다.
"""

import json
import os
from abc import ABC, abstractmethod

from ragent.models.parsed_message import NormalizedMessage


class BaseParser(ABC):
    """Transcript 파서 추상 베이스.

    각 어댑터(Claude Code, Codex, ...) 는 자신의 transcript 한 줄 스키마를
    NormalizedMessage 로 번역하는 서브클래스를 제공한다. 메시지 스트림 위에서
    도는 알고리즘(역순 1턴 추출, 전체 transcript turn 그룹핑)은 모든 파서가
    공유하는 Template Method 로 이 베이스에 둔다.
    """

    def __init__(self, transcript_path: str):
        self.transcript_path = transcript_path

    @abstractmethod
    def parse_transcript_line(self, json_line: dict) -> NormalizedMessage | None:
        """이 어댑터의 transcript 한 줄(json 파싱된 dict) 을 정규화 메시지로 변환.

        형식이 안 맞거나 RAGent 가 의미 있는 메시지로 해석하지 않는 줄(예: 시스템
        이벤트, 빈 본문) 은 None 을 반환한다.
        """

    def _should_skip_line(self, json_line: dict) -> bool:
        """parse_full_transcript 에서 raw 줄 단위로 호출되는 사전 필터 훅.

        parse_transcript_line 호출 이전에 줄을 통째로 버려야 하는 경우 True 를
        반환하도록 서브클래스가 오버라이드한다. 기본은 필터링하지 않음.

        parse_last_turn 은 이 훅을 호출하지 않는다 — 기존 parsing_modules.py 의
        역순 탐색이 tool_result 같은 줄도 turn 에 포함시켰던 동작을 그대로
        보존하기 위함.
        """
        return False

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
                    line = buffer[::-1].decode('utf-8')
                    buffer.clear()
                    if line.strip():
                        try:
                            data = json.loads(line)
                            parsed_message = self.parse_transcript_line(data)
                            if parsed_message:
                                turn.append(parsed_message)
                                if parsed_message.role == 'user' and parsed_message.content.startswith('[text]'):
                                    turn.reverse()
                                    return turn
                        except json.JSONDecodeError:
                            pass
                elif char != b'\n':
                    buffer.extend(char)

                position -= 1

        turn.reverse()
        return turn

    def parse_full_transcript(self) -> list[list[NormalizedMessage]]:
        """Transcript 전체를 정방향으로 읽어 turn 단위로 그룹핑하여 반환한다.

        각 turn 은 user '[text]' 메시지를 시작점으로 갖고, 다음 user '[text]'
        직전까지의 메시지들을 모은다. 첫 user '[text]' 이전의 메시지(있다면) 는
        어떤 turn 에도 속하지 않으므로 결과에 포함되지 않는다 — 기존
        session_end 인라인 로직과 동일한 동작.
        """
        if not os.path.exists(self.transcript_path):
            return []

        messages: list[NormalizedMessage] = []
        try:
            with open(self.transcript_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        line_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if self._should_skip_line(line_data):
                        continue
                    try:
                        msg = self.parse_transcript_line(line_data)
                    except Exception:
                        continue
                    if msg:
                        messages.append(msg)
        except OSError:
            return []

        turns: list[list[NormalizedMessage]] = []
        current: list[NormalizedMessage] = []
        for msg in messages:
            is_turn_start = (
                msg.role == "user"
                and msg.content.startswith("[text]")
            )
            if is_turn_start:
                if current:
                    turns.append(current)
                current = [msg]
            elif current:
                current.append(msg)
        if current:
            turns.append(current)

        return turns


class ClaudeCodeParser(BaseParser):
    """Claude Code transcript JSONL 파서.

    형식: 각 줄은 대략 다음과 같은 JSON 객체.
        {"message": {"role": ..., "content": ... or [...]}, "timestamp": ...}
    'message' 키가 없는 줄(시스템 이벤트) 은 무시한다.

    본 파서는 Phase 5 이전 ragent/modules/parsing_modules.py 의 _parse_line 로직과
    ragent/handlers/session_end.py 의 _is_tool_result_entry 헬퍼를 그대로
    이식한 것이다 — 동작이 1바이트도 달라지면 안 된다.
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
                        text_content += "[text]\n"
                        text_content += block.get('text', '') + "\n"
                    elif block_type == 'thinking':
                        text_content += "[thinking]\n"
                        text_content += block.get('thinking', '')
                    elif block_type == 'tool_use':
                        tool_name = block.get('name')
                        tool_input = block.get('input', {})
                        text_content += f"[{tool_name}]\n"
                        for val in tool_input.values():
                            text_content += f"{val}\n"

        if text_content.strip():
            return NormalizedMessage(
                role=role,
                content=text_content.strip(),
                timestamp=timestamp,
            )
        return None

    def _should_skip_line(self, json_line: dict) -> bool:
        # parse_full_transcript 경로에서만 호출되어 tool_result-only user 엔트리를
        # 줄 단위로 통째로 버린다. parse_last_turn 은 이 훅을 호출하지 않으므로
        # 기존 stop.py 동작은 영향받지 않는다.
        return self._is_tool_result_entry(json_line)

    @staticmethod
    def _is_tool_result_entry(line_data: dict) -> bool:
        msg = line_data.get("message")
        if not isinstance(msg, dict):
            return False
        if msg.get("role") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, list):
            return False
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )


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

    def _should_skip_line(self, json_line: dict) -> bool:
        # function_call_output 줄은 본문이 매우 클 수 있어 (긴 명령 출력, 대용량 파일
        # 읽기 결과 등) parse_full_transcript 에서 통째로 스킵한다. ClaudeCodeParser
        # 가 user-only-tool_result 줄을 _is_tool_result_entry 로 거르는 것과 같은
        # 동기. parse_last_turn 은 이 훅을 호출하지 않으므로 stop.py 처리 시점에
        # 가장 최근 tool_result 가 보존되는 동작은 ClaudeCodeParser 와 동일하게
        # 유지된다.
        if json_line.get("type") != "response_item":
            return False
        payload = json_line.get("payload") or {}
        if not isinstance(payload, dict):
            return False
        return payload.get("type") == "function_call_output"

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
