import pytest

from app.services.static_validator import FoofahStyleValidationError, StaticValidationError, validate_code_safety, validate_foofah_matrix_style


def test_valid_transform():
    code = """
import pandas as pd

def transform(df):
    result = df.copy()
    return result
"""
    validate_code_safety(code)


def test_missing_transform():
    with pytest.raises(StaticValidationError, match="transform"):
        validate_code_safety("def other(df):\n    return df")


@pytest.mark.parametrize("module", ["os", "subprocess", "requests"])
def test_forbidden_imports(module):
    code = f"import {module}\n\ndef transform(df):\n    return df"
    with pytest.raises(StaticValidationError):
        validate_code_safety(code)


@pytest.mark.parametrize("call", ["open('x')", "eval('1')", "exec('x=1')", "__import__('os')"])
def test_forbidden_calls(call):
    code = f"def transform(df):\n    {call}\n    return df"
    with pytest.raises(StaticValidationError):
        validate_code_safety(code)


def test_dunder_attribute_access():
    code = "def transform(df):\n    x = df.__class__\n    return df"
    with pytest.raises(StaticValidationError, match="Dunder"):
        validate_code_safety(code)


def test_disallowed_import_sklearn():
    code = "import sklearn\n\ndef transform(df):\n    return df"
    with pytest.raises(StaticValidationError, match="not allowed"):
        validate_code_safety(code)


def test_allowed_import_collections():
    code = "from collections import defaultdict\n\ndef transform(df):\n    groups = defaultdict(list)\n    return df"
    validate_code_safety(code)


def test_foofah_matrix_style_accepts_matrix_conversion():
    code = """
import pandas as pd

def transform(df):
    data = df.fillna('').astype(str).values.tolist()
    rows = [[row[0], row[-1]] for row in data]
    return pd.DataFrame(rows)
"""
    validate_code_safety(code)
    validate_foofah_matrix_style(code)


def test_foofah_matrix_style_accepts_conversion_from_cleaned_copy():
    code = """
import pandas as pd

def transform(df):
    result = df.copy()
    data = result.fillna('').astype(str).values.tolist()
    rows = [[row[0], row[-1]] for row in data]
    return pd.DataFrame(rows)
"""
    validate_code_safety(code)
    validate_foofah_matrix_style(code)


def test_foofah_matrix_style_requires_matrix_conversion():
    code = """
def transform(df):
    result = df.copy()
    return result
"""
    validate_code_safety(code)
    with pytest.raises(FoofahStyleValidationError, match="2D string matrix"):
        validate_foofah_matrix_style(code)


def test_foofah_matrix_style_rejects_synthetic_column_access():
    code = """
import pandas as pd

def transform(df):
    data = df.fillna('').astype(str).values.tolist()
    result = df[['col_0', 'col_5']]
    return result
"""
    validate_code_safety(code)
    with pytest.raises(FoofahStyleValidationError, match="synthetic pandas columns"):
        validate_foofah_matrix_style(code)


def test_foofah_matrix_style_rejects_numeric_conversion():
    code = """
import pandas as pd

def transform(df):
    data = df.fillna('').astype(str).values.tolist()
    rows = [[row[0], str(float(row[1]))] for row in data]
    return pd.DataFrame(rows)
"""
    validate_code_safety(code)
    with pytest.raises(FoofahStyleValidationError, match="preserve cells as strings"):
        validate_foofah_matrix_style(code)
