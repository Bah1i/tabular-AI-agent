import pandas as pd

from app.services.profiler import dataframe_profile, representative_sample


def _col(profile, name):
    return next(c for c in profile["columns"] if c["name"] == name)


def test_detects_number_date_category():
    df = pd.DataFrame(
        {
            "amount": [1, 2, 3, 4],
            "date": ["2025-01-10", "10.01.2025", "2025/01/11", "11.01.2025"],
            "category": ["a", "a", "b", "b"],
        }
    )
    profile = dataframe_profile(df)
    assert _col(profile, "amount")["semantic_type"] == "number"
    assert _col(profile, "date")["semantic_type"] == "date"
    assert _col(profile, "category")["semantic_type"] == "category"


def test_detects_null_ratio():
    profile = dataframe_profile(pd.DataFrame({"x": [1, None, None, 4]}))
    assert _col(profile, "x")["null_ratio"] == 0.5


def test_detects_duplicate_rows():
    profile = dataframe_profile(pd.DataFrame({"a": [1, 1], "b": ["x", "x"]}))
    assert profile["duplicate_rows"] == 1


def test_detects_outliers():
    profile = dataframe_profile(pd.DataFrame({"x": [1, 2, 3, 4, 1000]}))
    assert _col(profile, "x")["outlier_count"] > 0


def test_representative_sample_includes_head_tail_interesting_rows():
    df = pd.DataFrame({"x": list(range(100)), "nullable": [None] + list(range(1, 100))})
    sample = representative_sample(df, max_rows=12)
    values = set(sample["x"].tolist())
    assert 0 in values
    assert 99 in values
    assert len(sample) <= 12
