# https://code.claude.com/docs/ko/hooks#userpromptsubmit
# https://platform.claude.com/docs/ko/build-with-claude/prompt-engineering/claude-prompting-best-practices#xml

import logging
import os
import json

from ragent.models.chunk import Chunk
from ragent.modules.embedding_modules import HybridEmbedding
from ragent.modules.retrieval_modules import Retriever
from ragent.vectordb import QdrantStorage

logger = logging.getLogger("ragent")


def format_context_for_claude(chunks: list[Chunk]) -> str:
    """검색된 Chunk 리스트를 XML 형태의 문자열로 변환한다."""
    if not chunks:
        return ""

    context_str = "<context>\n"
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.file_path if chunk.metadata.file_path else f"snippet_{i+1}"
        
        context_str += f'<document index="{i+1}" source="{source}">\n'
        context_str += f'{chunk.payload}\n'
        context_str += "</document>\n"
    
    context_str += "</context>"
    return context_str


def handle(data: dict) -> None:
    """Index the last Q&A pair from the conversation."""
    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path", "")
    prompt = data.get("prompt", "")

    if not session_id:
        logger.warning("UserPromptSubmit: missing session_id")
        return
    
    if not transcript_path:
        logger.warning("Stop: missing transcript_path")
        return

    if not prompt:
        logger.warning("UserPromptSubmit: missing prompt")
        return
    
    embedder = HybridEmbedding()
    vectordb = QdrantStorage(os.path.basename(os.path.dirname(transcript_path)))
    retriever = Retriever(vectordb, embedder)

    search_results = retriever.retrieve(prompt)

    if search_results:
        logger.info(f"Retrieved {len(search_results)} relevant chunks.")
        formatted_context = format_context_for_claude(search_results)
        additional_context_msg = f"Retrieved relevant code:\n{formatted_context}"
    else:
        logger.info("No relevant chunks found.")
        additional_context_msg = "No relevant internal code snippets were found for this query in the vector database."
    
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional_context_msg
        }
    }
    print(json.dumps(output))