# ════════════════════════════════════════════════════════════════════════════
# core/document_loader.py - 문서 로더
# ════════════════════════════════════════════════════════════════════════════

import json
import os
from pathlib import Path
from typing import List, Dict
from config.settings import DEFAULT_ENCODING


class DocumentLoader:
    '''
    문서 로더 (.txt, .json)
    
    지원 형식:
    - .txt: 평문 파일
    - .json: JSON 형식
    '''
    
    def __init__(self, encoding: str = DEFAULT_ENCODING):
        self.encoding = encoding
        print(f"✅ DocumentLoader 초기화 (인코딩: {encoding})")
    
    def load_documents(
        self,
        directory_path: str
    ) -> List[Dict]:
        '''
        디렉토리의 모든 문서 로드
        
        Args:
            directory_path: 문서 디렉토리 경로
        
        Returns:
            문서 리스트
        '''
        
        documents = []
        path = Path(directory_path)
        
        if not path.exists():
            print(f"❌ 경로 없음: {directory_path}")
            return documents
        
        # .txt 파일 로드
        for txt_file in sorted(path.glob("*.txt")):
            doc_id = txt_file.stem
            
            with open(txt_file, 'r', encoding=self.encoding) as f:
                text = f.read()
            
            documents.append({
                "doc_id": doc_id,
                "text": text,
                "metadata": {
                    "source": "txt",
                    "file_path": str(txt_file),
                    "file_size": len(text)
                }
            })
            
            print(f"✅ 로드: {doc_id}.txt ({len(text)} 글자)")
        
        # .json 파일 로드
        for json_file in sorted(path.glob("*.json")):
            doc_id = json_file.stem
            
            with open(json_file, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    doc_id_item = f"{doc_id}_{i}"
                    text = item.get("text", "")
                    metadata = {k: v for k, v in item.items() if k != "text"}
                    metadata["source"] = "json"
                    
                    documents.append({
                        "doc_id": doc_id_item,
                        "text": text,
                        "metadata": metadata
                    })
            else:
                text = data.get("text", "")
                metadata = {k: v for k, v in data.items() if k != "text"}
                metadata["source"] = "json"
                
                documents.append({
                    "doc_id": doc_id,
                    "text": text,
                    "metadata": metadata
                })
            
            print(f"✅ 로드: {doc_id}.json")
        
        total_size = sum(len(d["text"]) for d in documents)
        print(f"\n✅ 총 {len(documents)}개 문서 로드 ({total_size} 글자)")
        
        return documents
    
    def load_single(self, file_path: str) -> Dict:
        '''단일 파일 로드'''
        
        path = Path(file_path)
        
        if not path.exists():
            print(f"❌ 파일 없음: {file_path}")
            return None
        
        if path.suffix == ".txt":
            with open(path, 'r', encoding=self.encoding) as f:
                text = f.read()
            
            return {
                "doc_id": path.stem,
                "text": text,
                "metadata": {"source": "txt", "file_path": str(path)}
            }
        
        elif path.suffix == ".json":
            with open(path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            text = data.get("text", "")
            metadata = {k: v for k, v in data.items() if k != "text"}
            metadata["source"] = "json"
            
            return {
                "doc_id": path.stem,
                "text": text,
                "metadata": metadata
            }
        
        else:
            print(f"❌ 지원하지 않는 형식: {path.suffix}")
            return None
