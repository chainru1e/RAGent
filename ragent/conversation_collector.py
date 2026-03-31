"""Claude Code JSONL 대화 파서 모듈.

사용자가 작성한 message_parser / parse_claude_history 파서와,
기존 transcript.py에서 가져온 Turn 클래스 및 유틸리티 함수를 결합한 모듈이다.

주요 구성:
    - Turn: 사용자 질문과 어시스턴트 응답 한 쌍을 담는 클래스
    - message_parser: JSONL 한 줄(딕셔너리)을 파싱하여 역할·내용·타임스탬프를 추출
    - parse_claude_history: JSONL 파일 전체를 읽어 message_parser로 파싱한 결과 리스트 반환
    - parse_transcript: parse_claude_history 결과를 user→assistant 쌍으로 묶어 Turn 리스트로 변환
    - get_last_turn: 트랜스크립트에서 마지막 Turn만 반환
    - get_session_summary_text: Turn 리스트를 요약 텍스트로 변환
"""

import json
from pathlib import Path


class Turn:
    """사용자-어시스턴트 간 한 번의 대화 교환(턴)을 표현하는 클래스.

    Claude Code 세션의 JSONL 트랜스크립트를 파싱하면, 사용자의 질문(user_prompt)과
    어시스턴트의 응답(assistant_response)이 한 쌍으로 묶여 Turn 객체가 된다.
    이 객체는 ChromaDB에 Q&A 쌍으로 인덱싱되거나, 세션 요약 생성에 사용된다.

    Attributes:
        user_prompt: 사용자가 입력한 질문 또는 명령 텍스트.
        assistant_response: 어시스턴트가 반환한 응답 텍스트.
                            사용자 메시지 다음에 어시스턴트 응답이 없는 경우 빈 문자열이 될 수 있다.
        user_uuid: JSONL 엔트리에 기록된 사용자 메시지의 고유 식별자.
                   ChromaDB에서 upsert 시 문서 ID로 활용된다.
                   값이 없으면 빈 문자열이 기본값이다.
        timestamp: 메시지가 생성된 시각(ISO 8601 형식).
                   값이 없으면 빈 문자열이 기본값이다.
        metadata: 추가 메타데이터를 담는 딕셔너리.
                  현재는 사용되지 않지만, 향후 확장을 위해 유지한다.
                  인스턴스 간 공유를 방지하기 위해 생성 시마다 새 딕셔너리가 할당된다.
    """

    def __init__(
        self,
        user_prompt: str,
        assistant_response: str,
        user_uuid: str = "",
        timestamp: str = "",
        metadata: dict | None = None,
    ) -> None:
        self.user_prompt = user_prompt
        self.assistant_response = assistant_response
        self.user_uuid = user_uuid
        self.timestamp = timestamp
        # metadata 기본값으로 None을 받고, None이면 새 딕셔너리를 생성한다.
        # 파이썬에서 가변 객체(dict)를 기본 인자로 직접 사용하면
        # 모든 인스턴스가 같은 딕셔너리를 공유하는 버그가 발생하므로 이렇게 처리한다.
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> dict:
        """Turn 객체를 일반 딕셔너리로 변환한다."""
        return {
            "user_prompt": self.user_prompt,
            "assistant_response": self.assistant_response,
            "user_uuid": self.user_uuid,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# JSONL 메시지 파서
# ---------------------------------------------------------------------------

def message_parser(json_line) -> dict:
    """
    jsonl에서 읽어들인 한 줄을 파싱하여 메시지를 추출한다
    Args:
        json_line (dict): JSON 데이터에서 파싱된 한 줄의 딕셔너리
    Returns:
        dict: 다음 키를 포함하는 딕셔너리:
              - 'timestamp': 메시지의 타임스탬프
              - 'role': 메시지의 역할 ('user' 또는 'assistant')
              - 'content': 추출되고 정제된 텍스트 내용
              내용이 없거나 파싱 대상이 아니면 빈 딕셔너리를 반환
    """

    conversation = {}

    # 시스템 이벤트 무시
    if 'message' not in json_line:
        return conversation

    msg_data = json_line['message']
    role = msg_data.get('role')
    content = msg_data.get('content')
    timestamp = json_line.get('timestamp')

    text_content = ""

    # User 메시지 처리
    if role == 'user':
        if isinstance(content, str):
            text_content += content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type')
                    if block_type == 'text':
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

    # 내용이 있으면 저장
    if text_content.strip():
        conversation = {
            'timestamp': timestamp,
            'role': role,
            'content': text_content.strip()
        }

    return conversation


def parse_claude_history(file_path):
    conversations = []

    with open(str(file_path), 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            conversation = message_parser(data)
            if conversation:
                conversations.append(conversation)

    return conversations


# ---------------------------------------------------------------------------
# 어댑터: Turn 기반 API (기존 transcript.py 호환 인터페이스)
# ---------------------------------------------------------------------------

def parse_transcript(path: str | Path) -> list[dict]:
    """JSONL 트랜스크립트 파일 전체를 파싱하여 딕셔너리 리스트로 반환한다.

    내부적으로 message_parser를 사용해 각 JSONL 라인을 파싱한 뒤,
    연속된 user → assistant 메시지를 한 쌍으로 묶는다.
    message_parser는 uuid를 반환하지 않으므로, 여기서 JSONL 원본 데이터에서
    uuid를 직접 추출하여 포함시킨다.

    Args:
        path: JSONL 트랜스크립트 파일 경로. 문자열 또는 Path 객체 모두 허용한다.

    Returns:
        딕셔너리의 리스트. 파일이 존재하지 않으면 빈 리스트를 반환한다.
        user 메시지 다음에 assistant 응답이 없으면 빈 응답으로 딕셔너리를 생성한다.
    """
    path = Path(path)
    if not path.exists():
        return []

    # JSONL 파일을 한 줄씩 읽어 message_parser로 파싱하고,
    # 원본 JSON에서 uuid도 함께 보관한다 (message_parser는 uuid를 반환하지 않으므로).
    messages: list[tuple[dict, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        conversation = message_parser(data)
        if conversation:
            messages.append((conversation, data.get("uuid", "")))

    # user → assistant 순서쌍을 Turn 객체로 변환한 뒤 딕셔너리로 반환한다.
    turns: list[dict] = []
    i = 0
    while i < len(messages):
        msg, uuid = messages[i]
        if msg["role"] == "user":
            assistant_text = ""
            if i + 1 < len(messages) and messages[i + 1][0]["role"] == "assistant":
                assistant_text = messages[i + 1][0]["content"]
                i += 2
            else:
                i += 1
            turns.append(Turn(
                user_prompt=msg["content"],
                assistant_response=assistant_text,
                user_uuid=uuid,
                timestamp=msg.get("timestamp", ""),
            ).to_dict())
        else:
            # user 없이 단독으로 나온 assistant 메시지는 건너뛴다
            i += 1

    return turns


def get_last_turn(path: str | Path) -> dict | None:
    """트랜스크립트에서 마지막 턴(사용자-어시스턴트 쌍)만 딕셔너리로 반환한다.

    Stop 훅에서 가장 최근 Q&A 쌍을 ChromaDB에 인덱싱할 때 사용된다.

    Args:
        path: JSONL 트랜스크립트 파일 경로.

    Returns:
        마지막 턴 딕셔너리. 트랜스크립트가 비어있거나 파일이 없으면 None을 반환한다.
    """
    turns = parse_transcript(path)
    return turns[-1] if turns else None


def get_session_summary_text(turns: list[dict]) -> str:
    """턴 딕셔너리 리스트를 사람이 읽을 수 있는 세션 요약 텍스트로 변환한다.

    SessionEnd 훅에서 전체 대화 내용을 요약하여 ChromaDB에 저장할 때 사용된다.
    각 턴의 질문은 최대 200자, 응답은 최대 300자까지만 포함한다.

    Args:
        turns: 턴 딕셔너리의 리스트.

    Returns:
        "Turn 1:\\nQ: ...\\nA: ..." 형식으로 연결된 요약 문자열.
        빈 리스트가 들어오면 빈 문자열을 반환한다.
    """
    if not turns:
        return ""

    lines = []
    for i, turn in enumerate(turns, 1):
        q = turn["user_prompt"][:200]
        a = turn["assistant_response"][:300]
        lines.append(f"Turn {i}:\nQ: {q}\nA: {a}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# CLI 진입점: 독립 실행 시 JSONL 파일을 파싱하여 텍스트 파일로 출력
# ---------------------------------------------------------------------------

def main():
    # 실행 예시
    file_path = 'b4145410-5e8f-4ae1-8148-d086411eb932.jsonl'
    # file_path = './tests/fixtures/sample_transcript.jsonl'
    parsed_log = parse_claude_history(file_path)

    output_file = 'parsed_conversation.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for log in parsed_log:
            f.write(f"[{log['timestamp']}] {log['role'].upper()}:\n")
            f.write(f"{log['content']}\n")
            f.write("=" * 60 + "\n")

    print(f"결과가 '{output_file}' 파일에 성공적으로 저장되었습니다.")


if __name__ == '__main__':
    main()
