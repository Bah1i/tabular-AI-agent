import json


SYSTEM_PROMPT = """
You are a safe Python/Pandas code generator for tabular data.

Return only a JSON object without Markdown. The JSON schema is:
{
  "code": "Python code as a string",
  "explanation": "short explanation in Russian",
  "plan": {
    "selected_columns": ["..."],
    "operations": ["..."],
    "parameters": {"key": "value"}
  }
}

Hard requirements:
1. Define function transform(df) that accepts pandas.DataFrame and returns pandas.DataFrame.
2. Start with result = df.copy().
3. Do not read files, write files, call network, call subprocess, or access environment variables.
4. Allowed imports: pandas as pd, numpy as np, re, math, datetime.
5. The output must match expected_output_rows exactly for transform mode.
6. Generate a general transformation for the full source table, not a hardcoded answer for the examples.
7. Be careful with mixed date formats, missing values, numbers stored as strings, duplicates, and column order.
8. Preserve the exact output column order implied by expected_output_rows or by the analytical query.
"""


FINAL_REMINDER = """
Final reminder: return only JSON with code, explanation, and plan. The code must define transform(df).
The transformation must generalize from the small expected example to the entire source table.
"""


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def build_generation_prompt(
    source_profile: dict,
    example_input: list[dict],
    example_output: list[dict],
    user_instruction: str | None = None,
    similar_examples: list[dict] | None = None,
) -> str:
    payload = {
        "mode": "transform_from_example",
        "task": (
            "Infer a reusable Pandas transform(df) function from a large source table and a small "
            "manually written expected output example."
        ),
        "important_validation_target": {
            "expected_output_rows": example_output,
            "rule": "transform(example_input_rows) must match expected_output_rows by columns, row count, and values.",
        },
        "user_instruction": user_instruction or "",
        "source_profile": source_profile,
        "example_input_rows": example_input,
        "similar_successful_transformations": similar_examples or [],
        "business_rules": [
            "Use the expected example to infer output columns, types, ordering, and required calculations.",
            "Do not hardcode row values from expected_output_rows.",
            "If dates are ambiguous, write robust parsing logic that handles common formats.",
            "If a value cannot be parsed safely, keep it as missing rather than inventing data.",
        ],
        "final_reminder": FINAL_REMINDER,
    }
    return _dump(payload)


def build_query_prompt(
    source_profile: dict,
    sample_rows: list[dict],
    user_instruction: str,
    similar_examples: list[dict] | None = None,
) -> str:
    payload = {
        "mode": "analytical_query",
        "task": "Generate a reusable Pandas transform(df) function that answers the user's analytical query.",
        "user_query": user_instruction,
        "source_profile": source_profile,
        "sample_rows": sample_rows,
        "similar_successful_transformations": similar_examples or [],
        "business_rules": [
            "Return a pandas.DataFrame, even if the query asks for one value.",
            "For top-N queries, sort deterministically and return the requested number of rows.",
            "Infer likely columns from names and semantic types, but do not invent columns.",
        ],
        "final_reminder": FINAL_REMINDER,
    }
    return _dump(payload)


def build_repair_prompt(
    previous_code: str,
    error_or_diff: str,
    source_profile: dict,
    example_input: list[dict],
    example_output: list[dict] | None = None,
    user_instruction: str | None = None,
    attempt_history: list[dict] | None = None,
    mode: str = "transform",
) -> str:
    payload = {
        "mode": mode,
        "task": "Repair the previous transform(df) function.",
        "critical_error_or_diff": error_or_diff,
        "previous_code": previous_code,
        "attempt_history": attempt_history or [],
        "user_instruction": user_instruction or "",
        "source_profile": source_profile,
        "example_input_rows": example_input,
        "expected_output_rows": example_output or [],
        "repair_rules": [
            "Fix the root cause, not only the shown row.",
            "Do not repeat a previous failed solution.",
            "Keep the solution general for the full table.",
            "Return only JSON with code, explanation, and plan.",
        ],
        "final_reminder": FINAL_REMINDER,
    }
    return _dump(payload)
