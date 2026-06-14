import json

import pandas as pd
from sqlalchemy import select

from app.models.job import TransformJob
from app.models.memory import TransformationMemory
from app.services.job_cache import prompt_cache_version
from app.services.transform_service import (
    _foofah_candidate_context,
    _foofah_failure_context,
    _foofah_memory_enabled,
    _memory_signature,
    _normalize_foofah_router_plan,
    _similar_transformations,
    _store_successful_memory,
)


def test_foofah_memory_enabled_reads_benchmark_context():
    disabled = json.dumps({"benchmark_context": {"memory_enabled": False}})
    enabled = json.dumps({"benchmark_context": {"memory_enabled": True}})

    assert _foofah_memory_enabled(disabled) is False
    assert _foofah_memory_enabled(enabled) is True
    assert _foofah_memory_enabled("plain instruction") is True


def test_foofah_failure_context_classifies_column_count_too_low_and_reroutes():
    diff = json.dumps(
        {
            "column_diffs": [
                {"actual": ["col_0"], "expected": ["col_0", "col_1", "col_2"]}
            ],
            "row_count": {},
            "value_diffs": [],
            "rule_violations": [],
        }
    )

    context = _foofah_failure_context(
        "Column mismatch",
        diff,
        [
            {"attempt": 1, "error": "Column mismatch", "code_hash": "aaa"},
            {"attempt": 2, "error": "Column mismatch", "code_hash": "bbb"},
        ],
        {"operator": "column_select"},
    )

    assert context["error_family"] == "column_count_too_low"
    assert context["actual_width"] == 1
    assert context["expected_width"] == 3
    assert context["failed_operator"] == "column_select"
    assert context["should_reroute"] is True


def test_foofah_candidate_context_is_extracted_from_instruction():
    instruction = """
Infer the FOOFAH table transformation.

{
  "expensive_candidate_mode": {
    "candidate_index": 2,
    "candidate_count": 3
  }
}
"""

    context = _foofah_candidate_context(instruction)

    assert context["enabled"] is True
    assert context["candidate_index"] == 2
    assert context["candidate_count"] == 3
    assert context["method"] == "independent_candidate_run"


def test_foofah_router_plan_uses_ranked_operator_for_candidate():
    plan = _normalize_foofah_router_plan(
        {
            "operator": "column_select",
            "ranked_operators": [
                {"operator": "column_select", "confidence": 0.9, "parameters": {}, "diversity_role": "primary"},
                {"operator": "wide_by_key", "confidence": 0.7, "parameters": {"key_col": 0}, "diversity_role": "alternative"},
            ],
        },
        {"enabled": True, "candidate_index": 2, "candidate_count": 3},
    )

    assert plan["operator"] == "wide_by_key"
    assert plan["parameters"] == {"key_col": 0}
    assert plan["candidate_selected_rank"] == 2
    assert plan["candidate_diversity_role"] == "alternative"


def test_foofah_reroute_avoids_failed_operator_when_ranked_alternative_exists():
    plan = _normalize_foofah_router_plan(
        {
            "operator": "record_pair_merge",
            "same_operator_allowed": False,
            "ranked_operators": [
                {"operator": "record_pair_merge", "confidence": 0.8, "parameters": {}},
                {"operator": "split_extract", "confidence": 0.6, "parameters": {}},
            ],
        },
        {"enabled": False},
        {"should_reroute": True, "failed_operator": "record_pair_merge"},
    )

    assert plan["operator"] == "split_extract"


def test_foofah_failure_context_detects_leading_column_swap():
    diff_json = json.dumps(
        {
            "column_diffs": [],
            "row_count": {},
            "value_diffs": [
                {"row": 0, "column": "col_0", "actual": "Product1", "expected": "9/1/2008"},
                {"row": 0, "column": "col_1", "actual": "9/1/2008", "expected": "Product1"},
            ],
            "rule_violations": [],
        }
    )

    context = _foofah_failure_context(
        "Value mismatch",
        diff_json,
        attempt_history=[],
        router_plan={"operator": "header_grid_to_long"},
    )

    assert context["error_family"] == "leading_column_swap"
    assert "swap first two output fields" in context["operator_suggestions"]
    assert "swap only the first two emitted fields" in context["direct_repair_instruction"]


def test_foofah_failure_context_detects_value_order_mismatch():
    diff_json = json.dumps(
        {
            "column_diffs": [],
            "row_count": {},
            "value_diffs": [
                {"row": 0, "column": "col_2", "actual": "60", "expected": "1,532,159"},
                {"row": 0, "column": "col_3", "actual": "1,532,159", "expected": "494,919"},
                {"row": 0, "column": "col_4", "actual": "76", "expected": "60"},
                {"row": 0, "column": "col_5", "actual": "494,919", "expected": "76"},
            ],
            "rule_violations": [],
        }
    )

    context = _foofah_failure_context(
        "Value mismatch",
        diff_json,
        attempt_history=[],
        router_plan={"operator": "stacked_metric_rows"},
    )

    assert context["error_family"] == "value_order_mismatch"
    assert "wide_by_key" in context["operator_suggestions"]
    assert "Keep extraction unchanged" in context["direct_repair_instruction"]


def test_foofah_memory_signature_uses_structure_not_column_names():
    left = pd.DataFrame([["A", "1"], ["A", "2"]], columns=["col_0", "col_1"])
    right = pd.DataFrame([["A", "1"], ["A", "2"]], columns=["name", "value"])

    assert _memory_signature(left, "foofah") == _memory_signature(right, "foofah")
    assert _memory_signature(left, "standard") != _memory_signature(right, "standard")


def test_foofah_memory_is_stored_only_when_explicitly_allowed(db_session):
    source_df = pd.DataFrame([["A", "1"]])
    job = TransformJob(
        source_filename="InputTable.csv",
        source_path="InputTable.csv",
        expected_path="OutputTable.csv",
        user_instruction="Infer FOOFAH",
        mode="transform",
        prompt_strategy="foofah",
        generated_code="def transform(df): return df",
        explanation="Visible example passed.",
    )

    _store_successful_memory(db_session, source_df, job, {"rows": 1})
    memories = list(db_session.scalars(select(TransformationMemory)).all())
    assert memories == []

    _store_successful_memory(db_session, source_df, job, {"rows": 1}, allow_foofah=True)
    memories = list(db_session.scalars(select(TransformationMemory)).all())
    assert len(memories) == 1
    assert memories[0].generated_code == job.generated_code
    assert memories[0].prompt_version == prompt_cache_version("foofah", cache_mode="honest")


def test_similar_transformations_are_scoped_to_prompt_version(db_session):
    source_df = pd.DataFrame([["A", "1"]])
    signature = _memory_signature(source_df, "foofah")
    db_session.add(
        TransformationMemory(
            source_columns_signature=signature,
            instruction="old",
            generated_code="def old(df): return df",
            success=True,
            prompt_version="old-version",
        )
    )
    db_session.add(
        TransformationMemory(
            source_columns_signature=signature,
            instruction="current",
            generated_code="def current(df): return df",
            success=True,
            prompt_version=prompt_cache_version("foofah", cache_mode="honest"),
        )
    )
    db_session.commit()

    examples = _similar_transformations(db_session, signature)

    assert [item["instruction"] for item in examples] == ["current"]
