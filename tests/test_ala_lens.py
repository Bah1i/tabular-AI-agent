from sqlalchemy import select

from app.models.ala_lens import AlaLensTypedDelta
from app.models.job import TransformJob
from app.services.ala_lens import lens_delta_statistics, parameter_model, record_lens_event


def test_parameter_model_adds_hybrid_lens_and_confidence_for_known_operator():
    parameter = parameter_model(
        "def transform(df):\n    return df\n",
        "Identity",
        {
            "operator": "prefix_suffix_chunks",
            "router_confidence": 0.8,
            "operations": ["prefix_suffix_chunks"],
            "output_rows_rule": "one output row per suffix chunk",
            "output_columns_rule": "prefix plus chunk",
            "value_order_rule": "preserve positional order",
            "self_check": "visible example matches",
        },
        validation_report={"column_diffs": [], "row_count": {}, "value_diffs": [], "rule_violations": []},
        event_type="stability",
    )

    assert parameter["hybrid_lens"]["status"] == "parameter_lens_backed"
    assert parameter["hybrid_lens"]["putback_scope"] == "parameter_constraints_only"
    assert parameter["p_ast_shadow"]["operator_family"] == "prefix_suffix_chunks"
    assert parameter["semantic_signature"]
    assert parameter["typed_view_delta"]["kind"] == "none"
    assert parameter["putback_policy"]["policy_name"] == "no_putback_needed"
    assert parameter["putback_mode"]["source_mutation"] == "forbidden"
    assert parameter["lens_law_checks"]["PutGet_runtime"] == "pass"
    assert parameter["restoration"]["level"] == "visible_view_restored"
    assert parameter["calibrated_confidence"]["visible_fit_score"] == 1.0
    assert parameter["calibrated_confidence"]["score"] > 0.7


def test_calibrated_confidence_uses_delta_penalties():
    parameter = parameter_model(
        "def transform(df):\n    return df\n",
        "Candidate",
        {"operator": "unknown", "router_confidence": 0.4},
        validation_report={
            "column_diffs": [],
            "row_count": {},
            "value_diffs": [{"row": 0, "column": "col_1", "actual": "a", "expected": "b"} for _ in range(8)],
            "rule_violations": [],
        },
        attempt_history=[{"attempt": 1, "success": False}],
        failure_context={"should_reroute": True},
        event_type="delta",
    )

    confidence = parameter["calibrated_confidence"]
    assert parameter["hybrid_lens"]["status"] == "operational_only"
    assert parameter["typed_view_delta"]["kind"] == "cell_value_delta"
    assert parameter["putback_policy"]["amendment_policy"] == "reroute"
    assert parameter["putback_mode"]["mode"] == "parameter_only_putback"
    assert parameter["lens_law_checks"]["PutGet_runtime"] == "fail"
    assert parameter["restoration"]["level"] == "not_restored"
    assert confidence["visible_fit_score"] < 1.0
    assert confidence["repair_instability_penalty"] > 0
    assert confidence["score"] < 0.5


def test_typed_delta_uses_failure_family_for_value_order_policy():
    parameter = parameter_model(
        "def transform(df):\n    return df\n",
        "Candidate",
        {"operator": "stacked_metric_rows", "value_order_rule": "grouped"},
        validation_report={
            "column_diffs": [],
            "row_count": {},
            "value_diffs": [{"row": 0, "column": "col_2", "actual": "703,255", "expected": "74"}],
            "rule_violations": [],
        },
        failure_context={
            "error_family": "value_order_mismatch",
            "direct_repair_instruction": "Keep extraction unchanged; change output ordering.",
            "should_reroute": False,
        },
        event_type="delta",
    )

    assert parameter["typed_view_delta"]["kind"] == "value_order_delta"
    assert parameter["putback_policy"]["policy_name"] == "adjust_value_order"
    assert "plan.value_order_rule" in parameter["putback_policy"]["target_fields"]


def test_record_lens_event_stores_typed_delta_row_and_statistics(db_session):
    job = TransformJob(source_filename="source.csv", source_path="source.csv", prompt_strategy="foofah")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    delta_parameter = parameter_model(
        "def transform(df):\n    return df\n",
        "Candidate",
        {"operator": "stacked_metric_rows"},
        validation_report={
            "column_diffs": [],
            "row_count": {},
            "value_diffs": [{"row": 0, "column": "col_2", "actual": "703,255", "expected": "74"}],
            "rule_violations": [],
        },
        failure_context={"error_family": "value_order_mismatch"},
        event_type="delta",
    )
    record_lens_event(
        db_session,
        job.id,
        1,
        "delta",
        prompt_strategy="foofah",
        parameter_before=delta_parameter,
        delta={"value_diffs": [{"row": 0, "column": "col_2", "actual": "703,255", "expected": "74"}]},
    )
    stable_parameter = parameter_model(
        "def transform(df):\n    return df\n",
        "Candidate",
        {"operator": "stacked_metric_rows"},
        validation_report={"column_diffs": [], "row_count": {}, "value_diffs": [], "rule_violations": []},
        event_type="stability",
    )
    record_lens_event(
        db_session,
        job.id,
        2,
        "stability",
        prompt_strategy="foofah",
        parameter_after=stable_parameter,
    )

    rows = list(db_session.scalars(select(AlaLensTypedDelta).where(AlaLensTypedDelta.job_id == job.id)).all())
    stats = lens_delta_statistics(db_session, [job.id])

    assert len(rows) == 2
    assert rows[0].delta_kind == "value_order_delta"
    assert rows[0].putback_policy_name == "adjust_value_order"
    assert stats["by_delta_kind"]["value_order_delta"]["restored"] == 1
    assert stats["restoration_success_rate_by_delta_family"]["value_order_delta"] == 1.0
    assert stats["putback_mode_summary"]["source_mutation_forbidden"] == 2
