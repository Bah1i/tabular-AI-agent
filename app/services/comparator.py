import math
import pandas as pd
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(); x.columns = [str(c) for c in x.columns]; return x.reset_index(drop=True)
def _values_equal(actual, expected, numeric_tol: float, date_tolerant: bool) -> bool:
    if pd.isna(actual) and pd.isna(expected): return True
    a_num = pd.to_numeric(pd.Series([actual]), errors='coerce').iloc[0]
    e_num = pd.to_numeric(pd.Series([expected]), errors='coerce').iloc[0]
    if not pd.isna(a_num) and not pd.isna(e_num):
        return math.isclose(float(a_num), float(e_num), rel_tol=numeric_tol, abs_tol=numeric_tol)
    if date_tolerant:
        a_dt = pd.to_datetime(pd.Series([actual]), errors='coerce').iloc[0]
        e_dt = pd.to_datetime(pd.Series([expected]), errors='coerce').iloc[0]
        if not pd.isna(a_dt) and not pd.isna(e_dt): return a_dt == e_dt
    return str(actual).strip() == str(expected).strip()
def compare_dataframes(actual: pd.DataFrame, expected: pd.DataFrame, numeric_tol: float = 1e-6, date_tolerant: bool = True) -> tuple[bool, str]:
    a = normalize_df(actual); e = normalize_df(expected)
    if list(a.columns) != list(e.columns): return False, f'Column mismatch. Actual={list(a.columns)}, expected={list(e.columns)}'
    if len(a) != len(e): return False, f'Row count mismatch. Actual={len(a)}, expected={len(e)}'
    diffs = []
    for i in range(len(e)):
        for col in e.columns:
            if not _values_equal(a.loc[i, col], e.loc[i, col], numeric_tol, date_tolerant):
                diffs.append({'row': int(i), 'column': col, 'actual': str(a.loc[i,col]), 'expected': str(e.loc[i,col])})
                if len(diffs) >= 20: return False, f'Value mismatch examples: {diffs}'
    return (False, f'Value mismatch examples: {diffs}') if diffs else (True, 'OK')
