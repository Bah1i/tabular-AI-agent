from types import SimpleNamespace

from app.services.llm_client import DeepSeekLLMClient


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def _client_with_fake_completion(content: str) -> tuple[DeepSeekLLMClient, FakeCompletions]:
    completions = FakeCompletions(content)
    client = DeepSeekLLMClient.__new__(DeepSeekLLMClient)
    client.model_name = "mock-model"
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_deepseek_generate_code_uses_passed_system_prompt():
    client, completions = _client_with_fake_completion(
        '{"code":"def transform(df):\\n    return df","explanation":"ok","plan":{}}'
    )

    client.generate_code("prompt", system_prompt="SPECIAL CODE SYSTEM")

    assert completions.calls[0]["messages"][0] == {"role": "system", "content": "SPECIAL CODE SYSTEM"}


def test_deepseek_generate_sql_uses_passed_system_prompt():
    client, completions = _client_with_fake_completion('{"sql":"SELECT 1","explanation":"ok"}')

    client.generate_sql("prompt", system_prompt="SPECIAL SQL SYSTEM")

    assert completions.calls[0]["messages"][0] == {"role": "system", "content": "SPECIAL SQL SYSTEM"}


def test_deepseek_generate_json_uses_passed_system_prompt():
    client, completions = _client_with_fake_completion('{"operator":"row_filter"}')

    client.generate_json("prompt", system_prompt="SPECIAL JSON SYSTEM")

    assert completions.calls[0]["messages"][0] == {"role": "system", "content": "SPECIAL JSON SYSTEM"}
