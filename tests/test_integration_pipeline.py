import pandas as pd

from app.models.job import JobStatus, TransformJob
from app.core.config import settings
from app.services.llm_client import JSONResult, LLMResult
from app.services.prompts import (
    FOOFAH_CODE_SYSTEM_PROMPT,
    FOOFAH_ROUTER_SYSTEM_PROMPT,
    GENERIC_TRANSFORM_SYSTEM_PROMPT,
)
from app.services.transform_service import run_transform_job


class DummyLLMClient:
    def __init__(self):
        self.code_system_prompts = []

    def generate_code(self, prompt: str, trace=None, generation_name: str = "generate_code", system_prompt: str | None = None) -> LLMResult:
        self.code_system_prompts.append(system_prompt)
        return LLMResult(
            code="""
import pandas as pd

def transform(df):
    result = df.copy()
    result["total"] = result["price"] * result["quantity"]
    return result[["product", "total"]]
""",
            explanation="Calculate total and keep product.",
            plan={
                "selected_columns": ["product", "price", "quantity"],
                "operations": ["calculate total", "select columns"],
                "parameters": {"total_formula": "price * quantity"},
            },
            raw_content="{}",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )


class FoofahDummyLLMClient:
    def __init__(self):
        self.code_system_prompts = []
        self.json_system_prompts = []

    def generate_json(self, prompt: str, trace=None, generation_name: str = "generate_json", system_prompt: str | None = None) -> JSONResult:
        self.json_system_prompts.append(system_prompt)
        return JSONResult(
            data={
                "operator": "identity_matrix_copy",
                "confidence": 0.9,
                "parameters": {},
                "ranked_operators": [{"operator": "identity_matrix_copy", "confidence": 0.9, "parameters": {}}],
            },
            raw_content="{}",
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
        )

    def generate_code(self, prompt: str, trace=None, generation_name: str = "generate_code", system_prompt: str | None = None) -> LLMResult:
        self.code_system_prompts.append(system_prompt)
        return LLMResult(
            code="""
import pandas as pd

def transform(df):
    data = df.fillna('').astype(str).values.tolist()
    return pd.DataFrame(data)
""",
            explanation="Copy the visible matrix exactly.",
            plan={"operations": ["identity_matrix_copy"], "parameters": {}},
            raw_content="{}",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )


def local_execute_code(code: str, input_df: pd.DataFrame, string_mode: bool = False) -> pd.DataFrame:
    namespace = {"pd": pd}
    exec(code, namespace)
    return namespace["transform"](input_df)


def test_pipeline_with_mock_llm_succeeds(monkeypatch, db_session, tmp_path):
    source = tmp_path / "source.csv"
    expected = tmp_path / "expected.csv"
    result_dir = tmp_path / "results"
    source.write_text("product,price,quantity\nA,10,2\nB,5,3\nC,7,4\n", encoding="utf-8")
    expected.write_text("product,total\nA,20\nB,15\n", encoding="utf-8")

    llm_client = DummyLLMClient()
    monkeypatch.setattr("app.services.transform_service.get_llm_client", lambda: llm_client)
    monkeypatch.setattr("app.services.transform_service.execute_code_in_sandbox", local_execute_code)
    monkeypatch.setattr(settings, "result_dir", str(result_dir))

    job = TransformJob(
        source_filename="source.csv",
        source_path=str(source),
        expected_path=str(expected),
        user_instruction="calculate total",
        mode="transform",
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    result = run_transform_job(db_session, job)

    assert result.status == JobStatus.success
    assert result.result_path is not None
    assert (result_dir / f"job_{job.id}_result.csv").exists()
    output = pd.read_csv(result.result_path)
    assert output.to_dict(orient="records") == [
        {"product": "A", "total": 20},
        {"product": "B", "total": 15},
        {"product": "C", "total": 28},
    ]
    assert llm_client.code_system_prompts == [GENERIC_TRANSFORM_SYSTEM_PROMPT]


def test_standard_transform_retries_full_source_when_head_validation_has_too_few_rows(monkeypatch, db_session, tmp_path):
    source = tmp_path / "parameterized_sales_source.xlsx"
    expected = tmp_path / "parameterized_sales_expected.xlsx"
    result_dir = tmp_path / "results"
    columns = [
        "Parameterized sales source",
        "Unnamed: 1",
        "Unnamed: 2",
        "Unnamed: 3",
        "Unnamed: 4",
        "parameter",
        "value",
    ]
    rows = [
        [None, None, None, None, None, "seasonal_discount", 0.07],
        [None, None, None, None, None, None, None],
        ["date", "product", "quantity", "price", "base_discount", "Formula", "total = quantity * price * (1 - discount)"],
        ["2025-03-01", "Laptop", 1, 100000, 0.05, None, None],
        ["2025-03-01", "Monitor", 2, 25000, 0.02, None, None],
        ["2025-03-02", "Keyboard", 4, 7000, 0, None, None],
        ["2025-03-02", "Mouse", 5, 1500, 0.01, None, None],
        ["2025-03-03", "Dock", 3, 12000, 0.03, None, None],
        ["2025-03-03", "Camera", 2, 9000, 0, None, None],
    ]
    pd.DataFrame(rows, columns=columns).to_excel(source, index=False)
    pd.DataFrame(
        [
            {"date": "2025-03-01", "product": "Laptop", "quantity": 1, "price": 100000, "effective_discount": 0.12, "total": 88000},
            {"date": "2025-03-01", "product": "Monitor", "quantity": 2, "price": 25000, "effective_discount": 0.09, "total": 45500},
            {"date": "2025-03-02", "product": "Keyboard", "quantity": 4, "price": 7000, "effective_discount": 0.07, "total": 26040},
            {"date": "2025-03-02", "product": "Mouse", "quantity": 5, "price": 1500, "effective_discount": 0.08, "total": 6900},
            {"date": "2025-03-03", "product": "Dock", "quantity": 3, "price": 12000, "effective_discount": 0.10, "total": 32400},
            {"date": "2025-03-03", "product": "Camera", "quantity": 2, "price": 9000, "effective_discount": 0.07, "total": 16740},
        ]
    ).to_excel(expected, index=False)

    class ParameterizedSalesLLM:
        def __init__(self):
            self.code_system_prompts = []

        def generate_code(self, prompt: str, trace=None, generation_name: str = "generate_code", system_prompt: str | None = None) -> LLMResult:
            self.code_system_prompts.append(system_prompt)
            return LLMResult(
                code="""
import pandas as pd

def transform(df):
    result = df.copy()
    seasonal_discount = float(result.loc[result['parameter'] == 'seasonal_discount', 'value'].iloc[0])
    header_idx = result[result.iloc[:, 0] == 'date'].index[0]
    data = result.loc[header_idx + 1:].copy()
    data.columns = result.iloc[header_idx].tolist()
    data = data[data['date'].notna() & (data['date'] != 'date')]
    data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce')
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    data['base_discount'] = pd.to_numeric(data['base_discount'], errors='coerce')
    data['effective_discount'] = data['base_discount'] + seasonal_discount
    data['total'] = data['quantity'] * data['price'] * (1 - data['effective_discount'])
    return data[['date', 'product', 'quantity', 'price', 'effective_discount', 'total']].reset_index(drop=True)
""",
                explanation="Extract parameter, scan all rows after the embedded header, calculate totals.",
                plan={
                    "selected_columns": ["parameter", "value", "date", "product", "quantity", "price", "base_discount"],
                    "operations": ["extract_parameter", "embedded_header", "calculate"],
                    "parameters": {"seasonal_discount": "source parameter row"},
                },
                raw_content="{}",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            )

    llm_client = ParameterizedSalesLLM()
    monkeypatch.setattr("app.services.transform_service.get_llm_client", lambda: llm_client)
    monkeypatch.setattr("app.services.transform_service.execute_code_in_sandbox", local_execute_code)
    monkeypatch.setattr(settings, "result_dir", str(result_dir))

    job = TransformJob(
        source_filename=source.name,
        source_path=str(source),
        expected_path=str(expected),
        user_instruction="calculate parameterized sales total",
        mode="transform",
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    result = run_transform_job(db_session, job)

    assert result.status == JobStatus.success
    output = pd.read_csv(result.result_path)
    assert len(output) == 6
    assert output["product"].tolist() == ["Laptop", "Monitor", "Keyboard", "Mouse", "Dock", "Camera"]


def test_simple_query_show_table_fast_path_avoids_llm(monkeypatch, db_session, tmp_path):
    source = tmp_path / "source.csv"
    result_dir = tmp_path / "results"
    source.write_text("product,price\nA,10\nB,5\n", encoding="utf-8")

    def fail_if_called():
        raise AssertionError("simple query should not call the LLM")

    monkeypatch.setattr("app.services.transform_service.get_llm_client", fail_if_called)
    monkeypatch.setattr(settings, "result_dir", str(result_dir))

    job = TransformJob(
        source_filename="source.csv",
        source_path=str(source),
        user_instruction="показать таблицу",
        mode="query",
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    result = run_transform_job(db_session, job)

    assert result.status == JobStatus.success
    assert result.attempts == 1
    output = pd.read_csv(result.result_path)
    assert output.to_dict(orient="records") == [
        {"product": "A", "price": 10},
        {"product": "B", "price": 5},
    ]


def test_foofah_pipeline_uses_foofah_router_and_code_system_prompts(monkeypatch, db_session, tmp_path):
    source = tmp_path / "InputTable.csv"
    expected = tmp_path / "OutputTable.csv"
    result_dir = tmp_path / "results"
    source.write_text("a\n1\n", encoding="utf-8")
    expected.write_text("a\n1\n", encoding="utf-8")

    llm_client = FoofahDummyLLMClient()
    monkeypatch.setattr("app.services.transform_service.get_llm_client", lambda: llm_client)
    monkeypatch.setattr("app.services.transform_service.execute_code_in_sandbox", local_execute_code)
    monkeypatch.setattr(settings, "result_dir", str(result_dir))

    job = TransformJob(
        source_filename="InputTable.csv",
        source_path=str(source),
        expected_path=str(expected),
        user_instruction="Infer FOOFAH",
        mode="transform",
        prompt_strategy="foofah",
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    result = run_transform_job(db_session, job)

    assert result.status == JobStatus.success
    assert llm_client.json_system_prompts == [FOOFAH_ROUTER_SYSTEM_PROMPT]
    assert llm_client.code_system_prompts == [FOOFAH_CODE_SYSTEM_PROMPT]
