import json
import os
import pathlib
import traceback
import importlib.util
import pandas as pd

WORKDIR = pathlib.Path(os.environ.get("SANDBOX_WORKDIR", "/work"))
INPUT_CSV = WORKDIR / "input.csv"
CODE_FILE = WORKDIR / "candidate.py"
OUTPUT_CSV = WORKDIR / "output.csv"
REPORT_JSON = WORKDIR / "report.json"

def write_report(ok: bool, error: str | None = None, rows: int | None = None, columns=None):
    REPORT_JSON.write_text(
        json.dumps(
            {
                "ok": ok,
                "error": error,
                "rows": rows,
                "columns": columns or [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

try:
    spec = importlib.util.spec_from_file_location("candidate", CODE_FILE)
    module = importlib.util.module_from_spec(spec)
    module.pd = pd
    spec.loader.exec_module(module)

    if not hasattr(module, "transform"):
        raise RuntimeError("Generated code must define function transform(df).")

    df = pd.read_csv(INPUT_CSV)
    result = module.transform(df)

    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("transform(df) must return pandas.DataFrame.")

    result.to_csv(OUTPUT_CSV, index=False)
    write_report(True, rows=len(result), columns=list(result.columns))

except Exception:
    write_report(False, error=traceback.format_exc())
