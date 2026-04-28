import pytest
from app.services.static_validator import validate_code_safety, StaticValidationError


def test_valid_code():
    code = '''
import pandas as pd

def transform(df):
    result = df.copy()
    return result
'''
    validate_code_safety(code)


def test_forbidden_import():
    code = '''
import os

def transform(df):
    return df
'''
    with pytest.raises(StaticValidationError):
        validate_code_safety(code)