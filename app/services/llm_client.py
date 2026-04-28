import json, time
from openai import OpenAI
from app.core.config import settings
from app.services.prompts import SYSTEM_PROMPT
class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    def generate_code(self, prompt: str, trace=None, generation_name: str = 'generate_code') -> tuple[str, str]:
        started = time.perf_counter()
        response = self.client.chat.completions.create(model=settings.deepseek_model, messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':prompt}], response_format={'type':'json_object'}, temperature=0.1)
        latency = time.perf_counter() - started
        content = response.choices[0].message.content or '{}'
        if trace:
            trace.generation(name=generation_name, model=settings.deepseek_model, input=prompt, output=content, metadata={'latency_seconds': latency})
        data = json.loads(content)
        return data['code'], data.get('explanation','')
