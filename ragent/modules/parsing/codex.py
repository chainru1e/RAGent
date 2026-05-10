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

import json

from ragent.models.parsed_message import NormalizedMessage
from ragent.modules.parsing.base import BaseParser


class CodexParser(BaseParser):
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
