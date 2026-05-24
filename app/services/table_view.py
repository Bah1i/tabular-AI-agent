from app.core.config import settings
from app.services.profiler import read_table


def table_preview(path: str, limit: int | None = None) -> dict:
    df = read_table(path)
    rows_limit = limit or settings.preview_rows
    sample = df.head(rows_limit)
    return {
        "rows_total": int(len(df)),
        "columns_total": int(len(df.columns)),
        "rows_shown": int(len(sample)),
        "columns": [str(c) for c in sample.columns],
        "rows": sample.where(sample.notna(), None).to_dict(orient="records"),
    }
