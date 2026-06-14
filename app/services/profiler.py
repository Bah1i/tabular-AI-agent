import json
import math
import pathlib

import pandas as pd


def _make_positional_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df


def read_table(path: str, headerless: bool = False) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        if headerless:
            return _make_positional_columns(pd.read_csv(path, header=None, dtype=str, keep_default_na=False))
        return pd.read_csv(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        if headerless:
            return _make_positional_columns(pd.read_excel(path, header=None, dtype=str, keep_default_na=False))
        return pd.read_excel(path)
    if lower.endswith(".json"):
        df = pd.read_json(path)
        return _make_positional_columns(df) if headerless else df
    if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
        df = pd.read_json(path, lines=True)
        return _make_positional_columns(df) if headerless else df
    if lower.endswith(".parquet"):
        df = pd.read_parquet(path)
        return _make_positional_columns(df) if headerless else df
    raise ValueError("Only CSV, XLSX, JSON, JSONL, and Parquet files are supported.")


def write_table(df: pd.DataFrame, path: str) -> None:
    lower = path.lower()
    if lower.endswith(".csv"):
        df.to_csv(path, index=False)
        return
    if lower.endswith(".json"):
        df.to_json(path, orient="records", force_ascii=False, indent=2)
        return
    if lower.endswith(".parquet"):
        df.to_parquet(path, index=False)
        return
    df.to_csv(path, index=False)


def _series_profile(series: pd.Series) -> dict:
    non_null = series.dropna()
    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(non_null) else 0.0
    try:
        parsed_dates = pd.to_datetime(non_null, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        parsed_dates = pd.to_datetime(non_null, errors="coerce", dayfirst=True)
    date_ratio = float(parsed_dates.notna().mean()) if len(non_null) else 0.0
    info = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "nulls": int(series.isna().sum()),
        "null_ratio": float(series.isna().mean()) if len(series) else 0.0,
        "unique": int(series.nunique(dropna=True)),
        "sample_values": [None if pd.isna(v) else str(v) for v in series.head(8).tolist()],
        "semantic_type": "text",
    }
    if numeric_ratio >= 0.8:
        info["semantic_type"] = "number"
        nums = numeric.dropna()
        if len(nums):
            q1 = float(nums.quantile(0.25))
            q3 = float(nums.quantile(0.75))
            iqr = q3 - q1
            outliers = nums[(nums < q1 - 1.5 * iqr) | (nums > q3 + 1.5 * iqr)] if iqr else nums.iloc[0:0]
            info.update({"min": float(nums.min()), "max": float(nums.max()), "mean": float(nums.mean()), "outlier_count": int(len(outliers))})
    elif date_ratio >= 0.8:
        info["semantic_type"] = "date"
        dates = parsed_dates.dropna()
        if len(dates):
            info.update({"min": str(dates.min().date()), "max": str(dates.max().date())})
    elif info["unique"] <= max(20, int(len(series) * 0.1)):
        info["semantic_type"] = "category"
        info["top_values"] = {str(k): int(v) for k, v in non_null.astype(str).value_counts().head(8).items()}
    return info


def representative_sample(df: pd.DataFrame, max_rows: int = 25) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    head_n = max(3, max_rows // 3)
    tail_n = max(2, max_rows // 5)
    sample_parts = [df.head(head_n), df.tail(tail_n)]
    interesting_indexes: set[int] = set()
    for col in df.columns:
        interesting_indexes.update(int(i) for i in df[df[col].isna()].head(2).index)
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() >= 5:
            interesting_indexes.update(int(i) for i in numeric.nlargest(2).index)
            interesting_indexes.update(int(i) for i in numeric.nsmallest(2).index)
    if interesting_indexes:
        sample_parts.append(df.loc[sorted(interesting_indexes)].head(max_rows))
    return pd.concat(sample_parts).drop_duplicates().head(max_rows).reset_index(drop=True)


def dataframe_profile(df: pd.DataFrame, max_rows: int = 10) -> dict:
    duplicate_rows = int(df.duplicated().sum()) if len(df) else 0
    columns = [_series_profile(df[col]) for col in df.columns]
    warnings: list[str] = []
    for col in columns:
        if col["null_ratio"] >= 0.3:
            warnings.append(f"Column {col['name']} has {col['null_ratio']:.0%} missing values.")
        if col.get("outlier_count", 0) > 0:
            warnings.append(f"Column {col['name']} has {col['outlier_count']} possible numeric outliers.")
    if duplicate_rows:
        warnings.append(f"Table has {duplicate_rows} duplicate rows.")
    return {
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": columns,
        "duplicate_rows": duplicate_rows,
        "warnings": warnings,
        "sample": representative_sample(df, max_rows=max_rows).to_dict(orient="records"),
    }


def dataframe_analysis(df: pd.DataFrame, max_rows: int = 10) -> dict:
    profile = dataframe_profile(df, max_rows=max_rows)
    critical: list[str] = []
    for col in profile["columns"]:
        if col["null_ratio"] >= 0.5:
            critical.append(f"Column {col['name']} is more than half empty.")
        if col["semantic_type"] == "number" and col.get("outlier_count", 0) > max(5, int(profile["rows"] * 0.02)):
            critical.append(f"Column {col['name']} has many possible outliers.")
    profile["summary"] = {
        "quality": "needs_attention" if critical else "ok",
        "critical_findings": critical,
        "human_readable": "The table needs attention: " + " ".join(critical) if critical else "The table structure looks usable for transformation.",
    }
    return profile


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes, list, dict, tuple)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def profile_to_json(profile: dict) -> str:
    return json.dumps(json_safe(profile), ensure_ascii=False, default=str)


def columns_signature(df: pd.DataFrame) -> str:
    return "|".join(f"{col}:{df[col].dtype}" for col in df.columns)


def file_extension(path: str) -> str:
    return pathlib.Path(path).suffix.lower()
