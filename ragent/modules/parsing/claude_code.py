"""Claude Code transcript JSONL 파서.

형식: 각 줄은 대략 다음과 같은 JSON 객체.
    {"message": {"role": ..., "content": ... or [...]}, "timestamp": ...}
'message' 키가 없는 줄(시스템 이벤트) 은 무시한다.

본 파서는 Phase 5 이전 ragent/modules/parsing_modules.py 의 _parse_line 로직과
ragent/handlers/session_end.py 의 _is_tool_result_entry 헬퍼를 그대로
이식한 것이다 — 동작이 1바이트도 달라지면 안 된다.
"""

from ragent.models.parsed_message import NormalizedMessage
from ragent.modules.parsing.base import BaseParser


class ClaudeCodeParser(BaseParser):
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
