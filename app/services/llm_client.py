import json
import time
from dataclasses import dataclass

from openai import OpenAI

from app.core.config import settings
from app.services.prompts import SYSTEM_PROMPT


@dataclass
class LLMResult:
    code: str
    explanation: str
    plan: dict
    raw_content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0


class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            (prompt_tokens / 1000) * settings.llm_input_1k_token_price_usd
            + (completion_tokens / 1000) * settings.llm_output_1k_token_price_usd
        )

    def generate_code(self, prompt: str, trace=None, generation_name: str = "generate_code") -> LLMResult:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        latency = time.perf_counter() - started
        content = response.choices[0].message.content or "{}"
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        estimated_cost = self._estimate_cost(prompt_tokens, completion_tokens)

        if trace:
            trace.generation(
                name=generation_name,
                model=settings.deepseek_model,
                input=prompt,
                output=content,
                metadata={
                    "latency_seconds": latency,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost,
                },
            )

        data = json.loads(content)
        return LLMResult(
            code=data["code"],
            explanation=data.get("explanation", ""),
            plan=data.get("plan", {}),
            raw_content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency,
            estimated_cost_usd=estimated_cost,
        )
