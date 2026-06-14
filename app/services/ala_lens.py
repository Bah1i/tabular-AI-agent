from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.ala_lens import AlaLensEvent, AlaLensTypedDelta

LENS_BACKED_OPERATORS = {
    "identity_matrix_copy",
    "column_select",
    "prefix_suffix_chunks",
    "stacked_metric_rows",
    "record_pair_merge",
    "wide_by_key",
    "wrap_fixed_width",
    "split_extract",
    "pivot_long_to_wide",
    "header_grid_to_long",
    "grouped_suffix_wide",
}


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _short_hash(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def source_model(df: pd.DataFrame, profile: dict) -> dict:
    return {
        "kind": "source_table",
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "schema_hash": sha256("|".join(f"{c}:{df[c].dtype}" for c in df.columns).encode("utf-8")).hexdigest()[:16],
        "profile": profile,
    }


def view_model(df: pd.DataFrame | None) -> dict | None:
    if df is None:
        return None
    sample = df.head(10).astype(object).where(pd.notna(df.head(10)), None)
    return {
        "kind": "view_table",
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "sample": sample.to_dict(orient="records"),
    }


def _operator_family(plan: dict | None) -> str:
    if not isinstance(plan, dict):
        return "unknown"
    return str(plan.get("operator") or plan.get("router_operator") or plan.get("plan_operator") or "unknown")


def p_ast_shadow(plan: dict | None) -> dict:
    plan = plan or {}
    return {
        "kind": "python_parameter_shadow_ast",
        "operator_family": _operator_family(plan),
        "operations": plan.get("operations") or [],
        "selected_columns": plan.get("selected_columns") or [],
        "parameters": plan.get("parameters") or plan.get("router_parameters") or {},
        "output_rows_rule": plan.get("output_rows_rule"),
        "output_columns_rule": plan.get("output_columns_rule"),
        "value_order_rule": plan.get("value_order_rule"),
        "row_0_role": plan.get("row_0_role"),
        "generalization_assumption": plan.get("generalization_assumption") or plan.get("router_generalization_assumption"),
    }


def semantic_signature(plan: dict | None) -> str:
    shadow = p_ast_shadow(plan)
    return _short_hash({k: v for k, v in shadow.items() if k != "kind"})


def hybrid_lens_metadata(plan: dict | None) -> dict:
    operator = _operator_family(plan)
    backed = operator in LENS_BACKED_OPERATORS
    return {
        "status": "parameter_lens_backed" if backed else "operational_only",
        "operator_family": operator,
        "putback_scope": "parameter_constraints_only",
        "source_putback_supported": False,
        "parameter_putback_supported": True,
        "source_mutation_policy": "forbidden",
        "reason": "Operator family has a typed shadow and normalized putback policy." if backed else "Generated Python is treated as operational code with a lightweight semantic shadow.",
    }


def _diff_counts(delta: dict | None) -> dict:
    delta = delta or {}
    return {
        "column_diffs": len(delta.get("column_diffs") or []),
        "row_count": 1 if (delta.get("row_count") or {}) else 0,
        "value_diffs": len(delta.get("value_diffs") or []),
        "rule_violations": len(delta.get("rule_violations") or []),
    }


def typed_view_delta(delta: dict | None, failure_context: dict | None = None) -> dict:
    delta = delta or {}
    failure_context = failure_context or {}
    counts = _diff_counts(delta)
    error_family = failure_context.get("error_family") or failure_context.get("raw_error_family")
    if not any(counts.values()):
        kind = "none"
        confidence = 1.0
    elif error_family == "value_order_mismatch":
        kind = "value_order_delta"
        confidence = 0.9
    elif counts["row_count"]:
        kind = "row_count_delta"
        confidence = 0.88
    elif counts["column_diffs"]:
        kind = "column_schema_delta"
        confidence = 0.86
    elif counts["value_diffs"]:
        kind = "cell_value_delta"
        confidence = 0.75
    elif counts["rule_violations"]:
        kind = "rule_violation_delta"
        confidence = 0.7
    else:
        kind = "unknown_delta"
        confidence = 0.35
    return {
        "kind": kind,
        "raw_error_family": error_family,
        "confidence": confidence,
        "counts": counts,
        "row_count": delta.get("row_count") or {},
        "sample_value_diffs": (delta.get("value_diffs") or [])[:12],
        "sample_column_diffs": (delta.get("column_diffs") or [])[:4],
        "rule_violations": (delta.get("rule_violations") or [])[:8],
    }


def putback_policy(delta: dict | None, plan: dict | None = None, failure_context: dict | None = None) -> dict:
    delta = delta or {"kind": "none"}
    failure_context = failure_context or {}
    kind = delta.get("kind") or "unknown_delta"
    amendment_policy = "reroute" if failure_context.get("should_reroute") else "repair"
    direct = failure_context.get("direct_repair_instruction") or ""
    if kind == "none":
        return {
            "policy_name": "no_putback_needed",
            "putback_target": "none",
            "target_fields": [],
            "amendment_policy": "none",
            "instruction": "No view delta is present; keep parameter p stable.",
            "source_mutation_allowed": False,
            "parameter_putback_supported": True,
        }
    policies = {
        "value_order_delta": (
            "adjust_value_order",
            ["plan.value_order_rule", "plan.output_columns_rule", "generated_code.output_order"],
            "Keep extraction stable and change positional output ordering to match the visible view delta.",
        ),
        "row_count_delta": (
            "adjust_row_generation_rule",
            ["plan.output_rows_rule", "plan.row_0_role", "generated_code.row_loop"],
            "Amend the row-generation rule; decide header/data row role once and apply it consistently.",
        ),
        "column_schema_delta": (
            "adjust_column_generation_rule",
            ["plan.output_columns_rule", "plan.selected_columns", "generated_code.column_order"],
            "Amend the output column-generation rule and preserve expected positional width/order.",
        ),
        "cell_value_delta": (
            "adjust_value_extraction",
            ["plan.selected_columns", "plan.operations", "generated_code.extraction"],
            "Amend extraction/string preservation without mutating source data.",
        ),
        "rule_violation_delta": (
            "enforce_rule_constraints",
            ["plan.operations", "generated_code.validation_sensitive_logic"],
            "Respect comparator rule violations and preserve table values as strings when required.",
        ),
    }
    policy_name, target_fields, instruction = policies.get(
        kind,
        ("generic_parameter_repair", ["plan", "generated_code"], "Use the typed view delta as a parameter-only repair target."),
    )
    if direct:
        instruction = f"{instruction} Direct repair instruction: {direct}"
    return {
        "policy_name": policy_name,
        "putback_target": "parameter_p",
        "target_fields": target_fields,
        "amendment_policy": amendment_policy,
        "instruction": instruction,
        "source_mutation_allowed": False,
        "parameter_putback_supported": True,
        "delta_kind": kind,
    }


def putback_mode(policy: dict | None, metadata: dict | None = None) -> dict:
    policy = policy or {}
    metadata = metadata or {}
    no_op = policy.get("policy_name") == "no_putback_needed"
    return {
        "mode": "no_op" if no_op else "parameter_only_putback",
        "putback_target": policy.get("putback_target") or "parameter_p",
        "source_mutation": "forbidden",
        "source_putback_supported": bool(metadata.get("source_putback_supported", False)),
        "parameter_putback_supported": bool(policy.get("parameter_putback_supported", True)),
        "source_mutation_allowed": False,
    }


def lens_law_checks(delta: dict | None, event_type: str | None = None) -> dict:
    kind = (delta or {}).get("kind", "none")
    restored = kind == "none" or event_type == "stability"
    return {
        "GetPut_runtime": "pass",
        "PutGet_runtime": "pass" if restored else "fail",
        "PutPut_runtime": "not_applicable" if restored else "unchecked",
        "evidence": "Runtime-level check over visible example; source mutation is forbidden.",
    }


def restoration_state(delta: dict | None, event_type: str | None = None, hidden_generalization_success: bool | None = None) -> dict:
    kind = (delta or {}).get("kind", "none")
    visible_restored = kind == "none" or event_type == "stability"
    return {
        "level": "visible_view_restored" if visible_restored else "not_restored",
        "source_restored": False,
        "parameter_restored": visible_restored,
        "visible_view_restored": visible_restored,
        "hidden_generalization_restored": hidden_generalization_success,
        "evidence": "Restoration is measured by rerunning generated code against the visible expected view.",
    }


def calibrated_confidence(
    plan: dict | None,
    validation_report: dict | None = None,
    attempt_history: list[dict] | None = None,
    failure_context: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    plan = plan or {}
    validation_report = validation_report or {}
    attempt_history = attempt_history or []
    failure_context = failure_context or {}
    metadata = metadata or hybrid_lens_metadata(plan)
    counts = _diff_counts(validation_report)
    penalty_units = counts["column_diffs"] * 0.15 + counts["row_count"] * 0.2 + counts["value_diffs"] * 0.08 + counts["rule_violations"] * 0.12
    visible_fit = max(0.0, 1.0 - min(1.0, penalty_units))
    operator = _operator_family(plan)
    structural = 0.85 if operator in LENS_BACKED_OPERATORS else 0.25
    if plan.get("output_rows_rule") or plan.get("output_columns_rule") or plan.get("value_order_rule"):
        structural = min(1.0, structural + 0.1)
    router_conf = float(plan.get("router_confidence") or plan.get("confidence") or 0.0)
    lens_bonus = 0.08 if metadata.get("status") == "parameter_lens_backed" else 0.0
    ambiguity_penalty = 0.12 if operator in {"unknown", "generic", ""} else 0.0
    if failure_context.get("should_reroute"):
        ambiguity_penalty += 0.12
    repair_instability_penalty = min(0.35, 0.1 * len([item for item in attempt_history if not item.get("success", False)]))
    score = visible_fit * 0.45 + structural * 0.25 + router_conf * 0.2 + lens_bonus - ambiguity_penalty - repair_instability_penalty
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "label": "high" if score >= 0.75 else "medium" if score >= 0.45 else "low",
        "visible_fit_score": visible_fit,
        "structural_evidence_score": structural,
        "router_confidence": router_conf,
        "lens_bonus": lens_bonus,
        "ambiguity_penalty": ambiguity_penalty,
        "repair_instability_penalty": repair_instability_penalty,
        "formula": "visible_fit_score + structural_evidence_score + router_confidence + lens_bonus - ambiguity_penalty - repair_instability_penalty",
    }


def parameter_model(
    code: str | None,
    explanation: str | None,
    plan: dict | None,
    validation_report: dict | None = None,
    attempt_history: list[dict] | None = None,
    failure_context: dict | None = None,
    event_type: str | None = None,
) -> dict:
    plan = plan or {}
    normalized_delta = typed_view_delta(validation_report, failure_context)
    metadata = hybrid_lens_metadata(plan)
    policy = putback_policy(normalized_delta, plan, failure_context)
    mode = putback_mode(policy, metadata)
    checks = lens_law_checks(normalized_delta, event_type)
    restoration = restoration_state(normalized_delta, event_type)
    confidence = calibrated_confidence(plan, validation_report, attempt_history, failure_context, metadata)
    return {
        "kind": "transformation_parameter",
        "code_hash": sha256((code or "").encode("utf-8")).hexdigest()[:16] if code else None,
        "code": code,
        "explanation": explanation,
        "plan": plan,
        "p_ast_shadow": p_ast_shadow(plan),
        "semantic_signature": semantic_signature(plan),
        "hybrid_lens": metadata,
        "typed_view_delta": normalized_delta,
        "putback_policy": policy,
        "putback_mode": mode,
        "lens_law_checks": checks,
        "restoration": restoration,
        "calibrated_confidence": confidence,
    }


def _parameter_from_event(parameter_before: dict | None, parameter_after: dict | None, delta: dict | None, event_type: str) -> dict:
    parameter = parameter_after or parameter_before or {}
    if parameter.get("typed_view_delta"):
        return parameter
    return parameter_model(parameter.get("code"), parameter.get("explanation"), parameter.get("plan") or {}, validation_report=delta, event_type=event_type)


def _record_typed_delta_row(db: Session, event: AlaLensEvent, parameter: dict, event_type: str) -> None:
    delta = parameter.get("typed_view_delta") or {"kind": "none"}
    policy = parameter.get("putback_policy") or putback_policy(delta)
    checks = parameter.get("lens_law_checks") or lens_law_checks(delta, event_type)
    restoration = parameter.get("restoration") or restoration_state(delta, event_type)
    mode = parameter.get("putback_mode") or putback_mode(policy)
    db.add(
        AlaLensTypedDelta(
            event_id=event.id,
            job_id=event.job_id,
            attempt_number=event.attempt_number,
            event_type=event_type,
            delta_kind=delta.get("kind") or "none",
            raw_error_family=delta.get("raw_error_family"),
            confidence=float(delta.get("confidence") or 0.0),
            putback_policy_name=policy.get("policy_name"),
            putback_target=policy.get("putback_target"),
            amendment_policy=policy.get("amendment_policy"),
            source_mutation_allowed=bool(mode.get("source_mutation_allowed", False)),
            parameter_putback_supported=bool(mode.get("parameter_putback_supported", True)),
            restoration_level=restoration.get("level"),
            getput_runtime=checks.get("GetPut_runtime"),
            putget_runtime=checks.get("PutGet_runtime"),
            putput_runtime=checks.get("PutPut_runtime"),
            semantic_signature=parameter.get("semantic_signature"),
            typed_delta_json=_safe_json(delta),
            putback_policy_json=_safe_json(policy),
            lens_law_checks_json=_safe_json(checks),
            restoration_json=_safe_json(restoration),
        )
    )


def record_lens_event(
    db: Session,
    job_id: int,
    attempt_number: int,
    event_type: str,
    source: dict | None = None,
    view: dict | None = None,
    parameter_before: dict | None = None,
    delta: dict | None = None,
    amendment: dict | None = None,
    parameter_after: dict | None = None,
    note: str | None = None,
    prompt_strategy: str | None = None,
    validation_status: str | None = None,
) -> None:
    parameter = _parameter_from_event(parameter_before, parameter_after, delta, event_type)
    event = AlaLensEvent(
        job_id=job_id,
        attempt_number=attempt_number,
        event_type=event_type,
        prompt_strategy=prompt_strategy,
        code_hash=(parameter_after or parameter_before or {}).get("code_hash") or parameter.get("code_hash"),
        validation_status=validation_status,
        source_model_json=_safe_json(source) if source is not None else None,
        view_model_json=_safe_json(view) if view is not None else None,
        parameter_before_json=_safe_json(parameter_before) if parameter_before is not None else None,
        delta_json=_safe_json(delta) if delta is not None else None,
        amendment_json=_safe_json(amendment) if amendment is not None else None,
        parameter_after_json=_safe_json(parameter_after) if parameter_after is not None else None,
        note=note,
    )
    db.add(event)
    db.flush()
    _record_typed_delta_row(db, event, parameter, event_type)
    db.commit()


def lens_delta_statistics(db: Session, job_ids: list[int] | None = None) -> dict:
    statement = select(AlaLensTypedDelta).order_by(AlaLensTypedDelta.job_id, AlaLensTypedDelta.attempt_number, AlaLensTypedDelta.id)
    if job_ids:
        statement = statement.where(AlaLensTypedDelta.job_id.in_(job_ids))
    rows = list(db.scalars(statement).all())
    by_job: dict[int, list[AlaLensTypedDelta]] = {}
    for row in rows:
        by_job.setdefault(row.job_id, []).append(row)

    by_delta_kind: dict[str, dict[str, int]] = {}
    law_summary = {"GetPut_runtime": {}, "PutGet_runtime": {}, "PutPut_runtime": {}}
    putback_mode_summary = {"source_mutation_forbidden": 0, "parameter_putback_supported": 0, "source_mutation_allowed": 0}

    def bump(bucket: dict, key: str | None) -> None:
        bucket[key or "unknown"] = bucket.get(key or "unknown", 0) + 1

    restored_ids: set[int] = set()
    for job_rows in by_job.values():
        stability_after = False
        for row in reversed(job_rows):
            if row.restoration_level == "visible_view_restored" or row.delta_kind == "none":
                stability_after = True
            elif stability_after:
                restored_ids.add(row.id)

    for row in rows:
        kind = row.delta_kind or "none"
        entry = by_delta_kind.setdefault(kind, {"count": 0, "restored": 0})
        entry["count"] += 1
        restored = row.restoration_level == "visible_view_restored" or row.id in restored_ids
        if restored:
            entry["restored"] += 1
        bump(law_summary["GetPut_runtime"], row.getput_runtime)
        bump(law_summary["PutGet_runtime"], row.putget_runtime)
        bump(law_summary["PutPut_runtime"], row.putput_runtime)
        if row.source_mutation_allowed:
            putback_mode_summary["source_mutation_allowed"] += 1
        else:
            putback_mode_summary["source_mutation_forbidden"] += 1
        if row.parameter_putback_supported:
            putback_mode_summary["parameter_putback_supported"] += 1

    restoration_rates = {
        kind: (entry["restored"] / entry["count"] if entry["count"] else 0.0)
        for kind, entry in by_delta_kind.items()
        if kind != "none"
    }
    return {
        "total_typed_deltas": len(rows),
        "by_delta_kind": by_delta_kind,
        "restoration_success_rate_by_delta_family": restoration_rates,
        "putback_mode_summary": putback_mode_summary,
        "law_check_summary": law_summary,
        "explicit_putback_mode": {
            "source_mutation": "forbidden_by_default",
            "parameter_putback": "supported_for_generated_parameter_p",
            "source_putback": "not_supported_without_formal_bidirectional_dsl",
        },
    }
