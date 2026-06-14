import pandas as pd

from app.services.comparator import compare_dataframes, compare_dataframes_report


def test_numeric_tolerance():
    ok, _ = compare_dataframes(pd.DataFrame({"x": [1.0000001]}), pd.DataFrame({"x": [1.0]}))
    assert ok


def test_date_tolerance():
    ok, _ = compare_dataframes(pd.DataFrame({"date": ["10.01.2025"]}), pd.DataFrame({"date": ["2025-01-10"]}))
    assert ok


def test_column_mismatch():
    ok, msg = compare_dataframes(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
    assert not ok
    assert "Column mismatch" in msg


def test_row_count_mismatch():
    ok, msg = compare_dataframes(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1]}))
    assert not ok
    assert "Row count mismatch" in msg


def test_value_mismatch():
    ok, msg = compare_dataframes(pd.DataFrame({"a": [2]}), pd.DataFrame({"a": [1]}))
    assert not ok
    assert "Value mismatch" in msg


def test_value_mismatch_displays_nan_as_missing():
    report = compare_dataframes_report(
        pd.DataFrame({"signup_date": [pd.NA]}),
        pd.DataFrame({"signup_date": ["2025-02-01"]}),
    )

    assert not report.ok
    assert report.details["value_diffs"][0]["actual"] == "<missing>"
    assert "nan" not in report.message.lower()


def test_required_column_rule():
    report = compare_dataframes_report(
        pd.DataFrame({"a": [1]}),
        pd.DataFrame({"a": [1]}),
        business_rules={"required_columns": ["a", "missing"]},
    )
    assert not report.ok
    assert report.details["rule_violations"][0]["rule"] == "required_column"


def test_unique_rule():
    report = compare_dataframes_report(
        pd.DataFrame({"id": [1, 1]}),
        pd.DataFrame({"id": [1, 1]}),
        business_rules={"columns": {"id": {"unique": True}}},
    )
    assert not report.ok
    assert report.details["rule_violations"][0]["rule"] == "unique"


def test_nullable_false_rule():
    report = compare_dataframes_report(
        pd.DataFrame({"name": ["a", None]}),
        pd.DataFrame({"name": ["a", None]}),
        business_rules={"columns": {"name": {"nullable": False}}},
    )
    assert not report.ok
    assert report.details["rule_violations"][0]["rule"] == "not_null"


def test_number_type_rule():
    report = compare_dataframes_report(
        pd.DataFrame({"amount": ["bad"]}),
        pd.DataFrame({"amount": ["bad"]}),
        business_rules={"columns": {"amount": {"type": "number"}}},
    )
    assert not report.ok
    assert report.details["rule_violations"][0]["rule"] == "number_type"


def test_date_type_rule():
    report = compare_dataframes_report(
        pd.DataFrame({"date": ["not-a-date"]}),
        pd.DataFrame({"date": ["not-a-date"]}),
        business_rules={"columns": {"date": {"type": "date"}}},
    )
    assert not report.ok
    assert report.details["rule_violations"][0]["rule"] == "date_type"


def test_exact_string_mode_disables_type_rules_for_foofah_style_tables():
    report = compare_dataframes_report(
        pd.DataFrame({"col_0": ["", "Anna"], "col_1": ["Math", "43"]}),
        pd.DataFrame({"col_0": ["", "Anna"], "col_1": ["Math", "43"]}),
        exact_strings=True,
    )
    assert report.ok
    assert report.details["rule_violations"] == []
