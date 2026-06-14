from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.generate_statistical_examples import (
    build_parameterized_sales_expected,
    build_sales_expected,
    build_sensors_expected,
    read_parameterized_sales_source,
    write_examples,
)


def test_generated_sales_expected_matches_source(tmp_path: Path):
    write_examples(tmp_path)
    source = pd.read_excel(tmp_path / "stat_sales_source.xlsx")
    expected = pd.read_excel(tmp_path / "stat_sales_expected.xlsx")
    recalculated = build_sales_expected(source)

    assert_frame_equal(expected, recalculated, check_dtype=False, atol=1e-6, rtol=1e-6)


def test_generated_sensors_expected_matches_source(tmp_path: Path):
    write_examples(tmp_path)
    source = pd.read_excel(tmp_path / "sensors_source.xlsx")
    expected = pd.read_excel(tmp_path / "sensors_expected.xlsx")
    recalculated = build_sensors_expected(source)

    assert_frame_equal(expected, recalculated, check_dtype=False, atol=1e-6, rtol=1e-6)


def test_generated_parameterized_sales_expected_matches_source_cell_parameter(tmp_path: Path):
    write_examples(tmp_path)
    source_rows, seasonal_discount = read_parameterized_sales_source(tmp_path / "parameterized_sales_source.xlsx")
    expected = pd.read_excel(tmp_path / "parameterized_sales_expected.xlsx")
    recalculated = build_parameterized_sales_expected(source_rows, seasonal_discount)

    assert seasonal_discount == 0.07
    assert_frame_equal(expected, recalculated, check_dtype=False, atol=1e-6, rtol=1e-6)
