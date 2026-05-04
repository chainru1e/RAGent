from ragent.config import LLM_API_BASE_URL
from openai import OpenAI

class LLMClient:
    def __init__(self, base_url=LLM_API_BASE_URL, system_prompt=""):
        
        self.client = OpenAI(base_url=base_url, api_key="sk-dummy")
        self.model_name = "qwen-local" # 로컬에서는 무시되지만 값은 필요
        
        self.system_prompt = system_prompt 

    def ask(self, prompt: str, override_system_prompt: str = None) -> str:
        # 일회성 시스템 프롬프트가 들어오면 그걸 쓰고, 아니면 원래 시스템 프롬프트(멤버 변수)을 쓴다.
        active_system_prompt = override_system_prompt if override_system_prompt else self.system_prompt
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": active_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"[Error] LLM 서버 통신 실패: {e}"