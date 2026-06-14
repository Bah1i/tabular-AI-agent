import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from openai import OpenAI

from app.core.config import settings
from app.services.prompts import GENERIC_JSON_SYSTEM_PROMPT, GENERIC_TRANSFORM_SYSTEM_PROMPT, SQL_SYSTEM_PROMPT


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


@dataclass
class SQLResult:
    sql: str
    explanation: str
    raw_content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0


@dataclass
class JSONResult:
    data: dict
    raw_content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0


def _loads_model_json(content: str) -> dict:
    """Parse model JSON while tolerating raw control characters inside code strings."""
    return json.loads(content, strict=False)


class BaseLLMClient(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate_code(
        self,
        prompt: str,
        trace=None,
        generation_name: str = "generate_code",
        system_prompt: str = GENERIC_TRANSFORM_SYSTEM_PROMPT,
    ) -> LLMResult:
        raise NotImplementedError

    @abstractmethod
    def generate_sql(
        self,
        prompt: str,
        trace=None,
        generation_name: str = "generate_sql",
        system_prompt: str = SQL_SYSTEM_PROMPT,
    ) -> SQLResult:
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        trace=None,
        generation_name: str = "generate_json",
        system_prompt: str = GENERIC_JSON_SYSTEM_PROMPT,
    ) -> JSONResult:
        raise NotImplementedError


class DeepSeekLLMClient(BaseLLMClient):
    provider_name = "deepseek"

    def __init__(self):
        self.model_name = settings.effective_llm_model
        self.client = OpenAI(api_key=settings.effective_llm_api_key, base_url=settings.effective_llm_base_url)

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            (prompt_tokens / 1000) * settings.llm_input_1k_token_price_usd
            + (completion_tokens / 1000) * settings.llm_output_1k_token_price_usd
        )

    def generate_code(
        self,
        prompt: str,
        trace=None,
        generation_name: str = "generate_code",
        system_prompt: str = GENERIC_TRANSFORM_SYSTEM_PROMPT,
    ) -> LLMResult:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
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
                model=self.model_name,
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

        data = _loads_model_json(content)
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

    def generate_sql(
        self,
        prompt: str,
        trace=None,
        generation_name: str = "generate_sql",
        system_prompt: str = SQL_SYSTEM_PROMPT,
    ) -> SQLResult:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
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
                model=self.model_name,
                input=prompt,
                output=content,
                metadata={"latency_seconds": latency, "total_tokens": total_tokens, "estimated_cost_usd": estimated_cost},
            )
        data = _loads_model_json(content)
        return SQLResult(
            sql=data["sql"],
            explanation=data.get("explanation", ""),
            raw_content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency,
            estimated_cost_usd=estimated_cost,
        )

    def generate_json(
        self,
        prompt: str,
        trace=None,
        generation_name: str = "generate_json",
        system_prompt: str = GENERIC_JSON_SYSTEM_PROMPT,
    ) -> JSONResult:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
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
                model=self.model_name,
                input=prompt,
                output=content,
                metadata={"latency_seconds": latency, "total_tokens": total_tokens, "estimated_cost_usd": estimated_cost},
            )
        return JSONResult(
            data=_loads_model_json(content),
            raw_content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency,
            estimated_cost_usd=estimated_cost,
        )


DeepSeekClient = DeepSeekLLMClient


def get_llm_client() -> BaseLLMClient:
    provider = (settings.llm_provider or "deepseek").strip().lower()
    if provider == "deepseek":
        return DeepSeekLLMClient()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
