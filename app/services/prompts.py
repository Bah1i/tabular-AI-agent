import json


SYSTEM_PROMPT = """
Ты — генератор безопасного Python/Pandas-кода для трансформации табличных данных.

Требования:
1. Верни только JSON без markdown.
2. JSON должен иметь поля:
   - code: строка с Python-кодом
   - explanation: краткое объяснение логики
3. Код должен содержать функцию transform(df), принимающую pandas.DataFrame и возвращающую pandas.DataFrame.
4. Разрешены только pandas и стандартные операции DataFrame.
5. Запрещены: os, sys, subprocess, socket, requests, pathlib, open, eval, exec, importlib, чтение/запись файлов, сетевые вызовы.
6. Не используй внешние файлы и интернет.
7. Не изменяй входной df напрямую: начни с result = df.copy().
"""


def build_generation_prompt(
    source_profile: dict,
    example_input: list[dict],
    example_output: list[dict],
    user_instruction: str | None = None,
) -> str:
    payload = {
        "task": "Синтезируй Python/Pandas-функцию transform(df) для преобразования таблицы.",
        "user_instruction": user_instruction or "",
        "source_profile": source_profile,
        "example_input_rows": example_input,
        "expected_output_rows": example_output,
        "validation_rule": "Результат transform(example_input_rows) должен совпасть с expected_output_rows по колонкам и значениям.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_repair_prompt(
    previous_code: str,
    error_or_diff: str,
    source_profile: dict,
    example_input: list[dict],
    example_output: list[dict],
    user_instruction: str | None = None,
) -> str:
    payload = {
        "task": "Исправь Python/Pandas-функцию transform(df).",
        "user_instruction": user_instruction or "",
        "previous_code": previous_code,
        "error_or_diff": error_or_diff,
        "source_profile": source_profile,
        "example_input_rows": example_input,
        "expected_output_rows": example_output,
        "requirements": "Верни только JSON с полями code и explanation.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)