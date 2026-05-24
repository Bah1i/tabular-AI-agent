import math
import json
import re
import pandas as pd


class ComparisonReport:
    def __init__(self, ok: bool, message: str, details: dict):
        self.ok = ok
        self.message = message
        self.details = details

    def to_json(self) -> str:
        return json.dumps(self.details, ensure_ascii=False, default=str)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(); x.columns = [str(c) for c in x.columns]; return x.reset_index(drop=True)
def _values_equal(actual, expected, numeric_tol: float, date_tolerant: bool) -> bool:
    if pd.isna(actual) and pd.isna(expected): return True
    a_num = pd.to_numeric(pd.Series([actual]), errors='coerce').iloc[0]
    e_num = pd.to_numeric(pd.Series([expected]), errors='coerce').iloc[0]
    if not pd.isna(a_num) and not pd.isna(e_num):
        return math.isclose(float(a_num), float(e_num), rel_tol=numeric_tol, abs_tol=numeric_tol)
    if date_tolerant:
        a_dt = _parse_date(actual)
        e_dt = _parse_date(expected)
        if not pd.isna(a_dt) and not pd.isna(e_dt): return a_dt == e_dt
    return str(actual).strip() == str(expected).strip()


def _parse_date(value):
    text = str(value).strip()
    if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", text):
        return pd.to_datetime(text, errors="coerce", yearfirst=True)
    return pd.to_datetime(text, errors="coerce", dayfirst=True)


def infer_rules_from_expected(expected: pd.DataFrame) -> dict:
    rules = {"required_columns": [str(c) for c in expected.columns], "columns": {}}
    for col in expected.columns:
        s = expected[col]
        col_rules = {
            "nullable": bool(s.isna().any()),
            "unique": bool(len(s) > 1 and s.nunique(dropna=True) == len(s.dropna())),
        }
        nums = pd.to_numeric(s.dropna(), errors="coerce")
        if len(nums) and nums.notna().mean() >= 0.8:
            col_rules["type"] = "number"
            if len(nums) >= 5:
                col_rules["min"] = float(nums.min())
                col_rules["max"] = float(nums.max())
        else:
            dates = pd.to_datetime(s.dropna(), errors="coerce", dayfirst=True)
            if len(dates) and dates.notna().mean() >= 0.8:
                col_rules["type"] = "date"
            else:
                col_rules["type"] = "text"
        rules["columns"][str(col)] = col_rules
    return rules


def validate_business_rules(df: pd.DataFrame, rules: dict | None) -> list[dict]:
    if not rules:
        return []
    violations = []
    for col in rules.get("required_columns", []):
        if col not in df.columns:
            violations.append({"rule": "required_column", "column": col, "message": "Column is missing."})
    for col, col_rules in rules.get("columns", {}).items():
        if col not in df.columns:
            continue
        s = df[col]
        if col_rules.get("nullable") is False and s.isna().any():
            violations.append({"rule": "not_null", "column": col, "count": int(s.isna().sum())})
        if col_rules.get("unique") and s.dropna().duplicated().any():
            violations.append({"rule": "unique", "column": col, "count": int(s.dropna().duplicated().sum())})
        if col_rules.get("type") == "number":
            nums = pd.to_numeric(s.dropna(), errors="coerce")
            bad = int(nums.isna().sum())
            if bad:
                violations.append({"rule": "number_type", "column": col, "count": bad})
            if "min" in col_rules and (nums < col_rules["min"]).any():
                violations.append({"rule": "min_range", "column": col, "min": col_rules["min"]})
            if "max" in col_rules and (nums > col_rules["max"]).any():
                violations.append({"rule": "max_range", "column": col, "max": col_rules["max"]})
        if col_rules.get("type") == "date":
            dates = pd.to_datetime(s.dropna(), errors="coerce", dayfirst=True)
            bad = int(dates.isna().sum())
            if bad:
                violations.append({"rule": "date_type", "column": col, "count": bad})
    return violations


def compare_dataframes_report(actual: pd.DataFrame, expected: pd.DataFrame, numeric_tol: float = 1e-6, date_tolerant: bool = True, business_rules: dict | None = None) -> ComparisonReport:
    a = normalize_df(actual); e = normalize_df(expected)
    details = {"column_diffs": [], "row_count": {}, "value_diffs": [], "rule_violations": []}
    if list(a.columns) != list(e.columns):
        details["column_diffs"].append({"actual": list(a.columns), "expected": list(e.columns)})
        return ComparisonReport(False, f'Column mismatch. Actual={list(a.columns)}, expected={list(e.columns)}', details)
    if len(a) != len(e):
        details["row_count"] = {"actual": len(a), "expected": len(e)}
        return ComparisonReport(False, f'Row count mismatch. Actual={len(a)}, expected={len(e)}', details)
    diffs = []
    for i in range(len(e)):
        for col in e.columns:
            if not _values_equal(a.loc[i, col], e.loc[i, col], numeric_tol, date_tolerant):
                diffs.append({'row': int(i), 'column': col, 'actual': str(a.loc[i,col]), 'expected': str(e.loc[i,col])})
                if len(diffs) >= 20:
                    details["value_diffs"] = diffs
                    return ComparisonReport(False, f'Value mismatch examples: {diffs}', details)
    rule_violations = validate_business_rules(a, business_rules or infer_rules_from_expected(e))
    details["value_diffs"] = diffs
    details["rule_violations"] = rule_violations
    if diffs:
        return ComparisonReport(False, f'Value mismatch examples: {diffs}', details)
    if rule_violations:
        return ComparisonReport(False, f'Business rule violations: {rule_violations[:10]}', details)
    return ComparisonReport(True, "OK", details)


def compare_dataframes(actual: pd.DataFrame, expected: pd.DataFrame, numeric_tol: float = 1e-6, date_tolerant: bool = True) -> tuple[bool, str]:
    report = compare_dataframes_report(actual, expected, numeric_tol=numeric_tol, date_tolerant=date_tolerant)
    return report.ok, report.message
