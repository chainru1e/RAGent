"""Tests for transcript parsing."""

from pathlib import Path

from ragent.transcript import get_last_turn, get_session_summary_text, parse_transcript

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_transcript.jsonl"


def test_parse_transcript_basic():
    turns = parse_transcript(SAMPLE)
    # Should have 3 valid turns:
    # 1. "What is Python?" -> answer
    # 2. "How do I install packages?" -> answer (thinking block excluded)
    # 3. "What is a virtual environment?" -> answer
    # Skipped: tool_result user message, <command-name> message
    assert len(turns) == 3


def test_parse_transcript_content():
    turns = parse_transcript(SAMPLE)
    assert turns[0].user_prompt == "What is Python?"
    assert "high-level programming language" in turns[0].assistant_response
    assert turns[0].user_uuid == "uuid-001"


def test_thinking_blocks_excluded():
    turns = parse_transcript(SAMPLE)
    # Second turn should not contain thinking text
    assert "Let me explain pip" not in turns[1].assistant_response
    assert "pip install" in turns[1].assistant_response


def test_tool_result_skipped():
    turns = parse_transcript(SAMPLE)
    prompts = [t.user_prompt for t in turns]
    assert not any("tool_result" in p for p in prompts)


def test_command_name_skipped():
    turns = parse_transcript(SAMPLE)
    prompts = [t.user_prompt for t in turns]
    assert not any("command-name" in p for p in prompts)


def test_get_last_turn():
    turn = get_last_turn(SAMPLE)
    assert turn is not None
    assert turn.user_prompt == "What is a virtual environment?"
    assert "isolated Python environment" in turn.assistant_response


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
