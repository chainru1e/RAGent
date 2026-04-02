import json

def parse_claude_history(file_path):
    conversations = []

    with open(file_path, 'r', encoding='utf-8') as f:
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

def message_parser(json_line: dict) -> dict:
    """
    jsonl에서 읽어들인 한 줄을 파싱하여 메시지를 추출한다.
    Args:
        json_line (dict): JSON 데이터에서 파싱된 한 줄의 딕셔너리.
    Returns:
        dict: 다음 키를 포함하는 딕셔너리:
              - 'timestamp': 메시지의 타임스탬프
              - 'role': 메시지의 역할 ('user' 또는 'assistant')
              - 'content': 추출되고 정제된 텍스트 내용
              - 'stop_reason': assistant 메시지의 생성 중단 사유
                               대화 턴의 종료 여부를 판단하는 핵심 기준(end_turn)
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
    stop_reason = msg_data.get('stop_reason')

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
        conversation = {
            'timestamp': timestamp,
            'role': role,
            'content': text_content.strip(),
            'stop_reason': stop_reason
        }

    return conversation

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