"""Tests for conversation_collector parsing."""

from pathlib import Path

from ragent.conversation_collector import (
    get_last_turn,
    get_session_summary_text,
    message_parser,
    parse_transcript,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_transcript.jsonl"


# ---------------------------------------------------------------------------
# parse_transcript tests
# ---------------------------------------------------------------------------

def test_parse_transcript_basic():
    turns = parse_transcript(SAMPLE)
    # 5 turns:
    # 1. "What is Python?" -> answer
    # 2. "How do I install packages?" -> answer (thinking + text)
    # 3. tool_result user -> tool_use + text assistant
    # 4. "<command-name>help</command-name>" -> empty (next entry is user)
    # 5. "What is a virtual environment?" -> answer
    assert len(turns) == 5


def test_parse_transcript_content():
    turns = parse_transcript(SAMPLE)
    assert turns[0]["user_prompt"] == "What is Python?"
    assert "high-level programming language" in turns[0]["assistant_response"]
    assert turns[0]["user_uuid"] == "uuid-001"


def test_thinking_blocks_included():
    turns = parse_transcript(SAMPLE)
    # Second turn includes [thinking] marker and thinking content
    assert "[thinking]" in turns[1]["assistant_response"]
    assert "Let me explain pip" in turns[1]["assistant_response"]
    assert "pip install" in turns[1]["assistant_response"]


def test_tool_result_included():
    turns = parse_transcript(SAMPLE)
    prompts = [t["user_prompt"] for t in turns]
    assert any("[tool_result]" in p for p in prompts)


def test_command_name_included():
    turns = parse_transcript(SAMPLE)
    prompts = [t["user_prompt"] for t in turns]
    assert any("command-name" in p for p in prompts)


def test_get_last_turn():
    turn = get_last_turn(SAMPLE)
    assert turn is not None
    assert turn["user_prompt"] == "What is a virtual environment?"
    assert "isolated Python environment" in turn["assistant_response"]


def test_get_last_turn_nonexistent():
    turn = get_last_turn("/nonexistent/path.jsonl")
    assert turn is None


def test_get_session_summary_text():
    turns = parse_transcript(SAMPLE)
    summary = get_session_summary_text(turns)
    assert "Turn 1:" in summary
    assert "What is Python?" in summary
    assert len(summary) > 0


def test_get_session_summary_empty():
    summary = get_session_summary_text([])
    assert summary == ""


# ---------------------------------------------------------------------------
# message_parser unit tests
# ---------------------------------------------------------------------------

def test_message_parser_user_string():
    entry = {
        "message": {"role": "user", "content": "Hello world"},
        "timestamp": "2026-01-01T00:00:00Z",
    }
    result = message_parser(entry)
    assert result["role"] == "user"
    assert result["content"] == "Hello world"
    assert result["timestamp"] == "2026-01-01T00:00:00Z"


def test_message_parser_assistant_blocks():
    entry = {
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "deep thought"},
                {"type": "text", "text": "visible answer"},
                {"type": "tool_use", "name": "grep", "input": {"pattern": "foo"}},
            ],
        },
        "timestamp": "2026-01-01T00:00:01Z",
    }
    result = message_parser(entry)
    assert result["role"] == "assistant"
    assert "[thinking]" in result["content"]
    assert "deep thought" in result["content"]
    assert "visible answer" in result["content"]
    assert "[grep]" in result["content"]
    assert "foo" in result["content"]


def test_message_parser_no_message():
    entry = {"type": "system", "timestamp": "2026-01-01T00:00:00Z"}
    result = message_parser(entry)
    assert result == {}
