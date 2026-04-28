import pandas as pd
from app.services.comparator import compare_dataframes
def test_numeric_tolerance():
    ok, _ = compare_dataframes(pd.DataFrame({'x':[1.0000001]}), pd.DataFrame({'x':[1.0]})); assert ok
def test_date_tolerance():
    ok, _ = compare_dataframes(pd.DataFrame({'date':['10.01.2025']}), pd.DataFrame({'date':['2025-01-10']})); assert ok
