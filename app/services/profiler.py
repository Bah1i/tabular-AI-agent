import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError("Only CSV and XLSX files are supported.")


def dataframe_profile(df: pd.DataFrame, max_rows: int = 10) -> dict:
    return {
        "rows": int(len(df)),
        "columns": [
            {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "nulls": int(df[col].isna().sum()),
                "sample_values": [None if pd.isna(v) else str(v) for v in df[col].head(5).tolist()],
            }
            for col in df.columns
        ],
        "sample": df.head(max_rows).to_dict(orient="records"),
    }