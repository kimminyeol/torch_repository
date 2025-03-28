# extractor/utils.py

from openai import OpenAI
import time
from string import Template
import os 
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_prompt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def call_openai(prompt: str, model="gpt-3.5-turbo") -> str:
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error (attempt {i+1}): {e}")
            time.sleep(2)
    raise RuntimeError("GPT 호출 실패")