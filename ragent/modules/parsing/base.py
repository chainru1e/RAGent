"""Transcript 파서 추상 베이스.

각 어댑터(Claude Code, Codex, ...) 는 자신의 transcript 한 줄 스키마를
NormalizedMessage 로 번역하는 서브클래스를 제공한다. 메시지 스트림 위에서
도는 알고리즘(역순 1턴 추출, 전체 transcript turn 그룹핑)은 모든 파서가
공유하는 Template Method 로 이 베이스에 둔다.
"""

import json
import os
from abc import ABC, abstractmethod

from ragent.models.parsed_message import NormalizedMessage


class BaseParser(ABC):
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
