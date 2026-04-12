import os
import json

class MessageParser:
    def __init__(self, path: str):
        self.path = path

    def _parse_line(self, json_line: dict) -> dict:
        """
        jsonl에서 읽어들인 한 줄을 파싱하여 메시지를 추출한다.
        Args:
            json_line (dict): JSON 데이터에서 파싱된 한 줄의 딕셔너리.
        Returns:
            dict: 다음 키를 포함하는 딕셔너리:
                - 'timestamp': 메시지의 타임스탬프
                - 'role': 메시지의 역할 ('user' 또는 'assistant')
                - 'content': 추출되고 정제된 텍스트 내용
                내용이 없거나 파싱 대상이 아니면 빈 딕셔너리를 반환
        """

        parsed_message = {}

        # 시스템 이벤트 무시
        if 'message' not in json_line:
            return parsed_message
        
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
                        
        # 내용이 있으면 저장
        if text_content.strip():
            parsed_message = {
                'timestamp': timestamp,
                'role': role,
                'content': text_content.strip()
            }

        return parsed_message
    
    def parse_last_turn(self) -> list:
        """
        jsonl 파일을 역순으로 읽어 가장 최근의 대화 1턴을 추출한다.
        
        파일의 끝에서부터 바이트 단위로 거꾸로 읽어 올라가며 한 줄씩 파싱하고,
        사용자(user)의 실제 텍스트 입력('[text]')을 턴의 시작점으로 간주하여 탐색을 종료한다.

        Returns:
            turn: 파싱된 메시지 딕셔너리들의 리스트.
                - 역순으로 읽어 차례대로 append 한 뒤 순서를 복원하므로, 리스트의 첫 번째 요소(인덱스 0)가
                  턴을 시작한 user의 메시지이며 마지막 요소가 가장 최신 메시지가 됨.
                - 파일이 존재하지 않거나 내용이 비어있을 경우 빈 리스트([])를 반환.
        """
        turn = []

        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return turn
        
        with open(self.path, 'rb') as f:
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
                            parsed_message = self._parse_line(data)
                            if parsed_message:
                                turn.append(parsed_message)
                                if parsed_message['role'] == 'user' and parsed_message['content'].startswith('[text]'):
                                    turn.reverse()
                                    return turn
                        except json.JSONDecodeError:
                            pass
                elif char != b'\n':
                    buffer.extend(char)

                position -= 1
        
        turn.reverse()
        return turn