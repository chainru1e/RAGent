# ════════════════════════════════════════════════════════════════════════════
# core/chunking.py - 문서 청킹
# ════════════════════════════════════════════════════════════════════════════

import re
from typing import List, Dict
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_STRATEGY


class DocumentChunker:
    '''
    의미론적 문서 청킹
    
    특징:
    - 문장 단위 분할
    - 의미 컨텍스트 보존
    - 겹침 처리
    '''
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        strategy: str = CHUNKING_STRATEGY
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        print(f"✅ DocumentChunker 초기화")
        print(f"   전략: {strategy}")
        print(f"   청크 크기: {chunk_size}자")
        print(f"   겹침: {chunk_overlap}자")
    
    def chunk_text(
        self,
        text: str,
        doc_id: str
    ) -> List[Dict]:
        '''
        텍스트를 청크로 분할
        
        Args:
            text: 입력 텍스트
            doc_id: 문서 ID
        
        Returns:
            청크 리스트
        '''
        
        if self.strategy == "semantic":
            return self._semantic_chunk(text, doc_id)
        elif self.strategy == "fixed":
            return self._fixed_chunk(text, doc_id)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _semantic_chunk(self, text: str, doc_id: str) -> List[Dict]:
        '''의미론적 청킹 (문장 단위)'''
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_num = 0
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.chunk_size and current_chunk:
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{chunk_num}",
                    "original_doc_id": doc_id,
                    "chunk_num": chunk_num,
                    "text": current_chunk.strip()
                })
                chunk_num += 1
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        if current_chunk:
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_num}",
                "original_doc_id": doc_id,
                "chunk_num": chunk_num,
                "text": current_chunk.strip()
            })
        
        return chunks
    
    def _fixed_chunk(self, text: str, doc_id: str) -> List[Dict]:
        '''고정 크기 청킹 (겹침 포함)'''
        
        chunks = []
        chunk_num = 0
        step = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]
            
            if chunk_text.strip():
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{chunk_num}",
                    "original_doc_id": doc_id,
                    "chunk_num": chunk_num,
                    "text": chunk_text.strip()
                })
                chunk_num += 1
        
        return chunks
