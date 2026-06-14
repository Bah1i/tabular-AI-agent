import json
import pathlib
import re
import time
from datetime import datetime
from hashlib import sha256

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.attempt import TransformAttempt
from app.models.job import TransformJob, JobStatus
from app.models.memory import TransformationMemory
from app.models.metric import JobMetric
from app.services.ala_lens import parameter_model, putback_mode, putback_policy, record_lens_event, source_model, typed_view_delta, view_model
from app.services.comparator import compare_dataframes_report, infer_rules_from_expected
from app.services.job_cache import fill_job_cache_keys, prompt_cache_version
from app.services.llm_client import LLMResult, JSONResult, get_llm_client
from app.services.profiler import (
    columns_signature,
    dataframe_profile,
    profile_to_json,
    read_table,
    representative_sample,
)
from app.services.prompts import (
    FOOFAH_CODE_SYSTEM_PROMPT,
    FOOFAH_ROUTER_SYSTEM_PROMPT,
    GENERIC_TRANSFORM_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_foofah_generation_prompt,
    build_foofah_operator_prompt,
    build_foofah_router_prompt,
    build_generation_prompt,
    build_query_prompt,
    build_repair_prompt,
)
from app.services.sandbox_executor import execute_code_in_sandbox
from app.services.static_validator import validate_code_safety, validate_foofah_matrix_style
from app.services.tracing import score_trace, trace_job, trace_span, update_trace


def _save_result(job_id: int, df: pd.DataFrame) -> str:
    pathlib.Path(settings.result_dir).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(settings.result_dir) / f"job_{job_id}_result.csv"
    df.to_csv(p, index=False)
    return str(p)


def _as_positional_table(df: pd.DataFrame) -> pd.DataFrame:
    positional = df.copy()
    positional.columns = [f"col_{i}" for i in range(len(positional.columns))]
    return positional


def _short_hash(value: str | None) -> str | None:
    if not value:
        return None
    return sha256(value.encode("utf-8")).hexdigest()[:16]


def _plan_signature(plan: dict | None) -> str | None:
    if not plan:
        return None
    return _short_hash(json.dumps(plan, ensure_ascii=False, sort_keys=True, default=str))


def _foofah_memory_signature(source_df: pd.DataFrame) -> str:
    data = source_df.fillna("").astype(str).values.tolist()
    row_patterns: list[str] = []
    for row in data[:25]:
        cells = [str(cell) for cell in row]
        nonempty_positions = ",".join(str(i) for i, cell in enumerate(cells) if cell.strip())
        markers = "".join(
            [
                ":" if any(":" in cell for cell in cells) else "",
                "=" if any("=" in cell for cell in cells) else "",
                "," if any("," in cell for cell in cells) else "",
                "*" if any(cell.strip().startswith("*") for cell in cells) else "",
            ]
        )
        row_patterns.append(f"{nonempty_positions}:{markers or '-'}")
    raw = f"foofah|rows={len(source_df)}|cols={len(source_df.columns)}|patterns={'|'.join(row_patterns)}"
    return f"foofah:{_short_hash(raw)}"


def _memory_signature(source_df: pd.DataFrame, prompt_strategy: str | None) -> str:
    if prompt_strategy == "foofah":
        return _foofah_memory_signature(source_df)
    return columns_signature(source_df)


def _json_objects_from_text(text: str | None) -> list[dict]:
    if not text:
        return []
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
        index = start + max(end, 1)
    return objects


def _foofah_candidate_context(user_instruction: str | None) -> dict:
    for item in _json_objects_from_text(user_instruction):
        mode = item.get("expensive_candidate_mode")
        if not isinstance(mode, dict):
            continue
        candidate_index = int(mode.get("candidate_index") or 1)
        candidate_count = int(mode.get("candidate_count") or 1)
        return {
            "enabled": candidate_count > 1,
            "candidate_index": candidate_index,
            "candidate_count": candidate_count,
            "method": "independent_candidate_run",
            "explanation": (
                "Дорогой FOOFAH-режим запускает несколько независимых кандидатов для одного case. "
                "У каждого кандидата свой router/code/repair loop; benchmark выбирает первый вариант, "
                "который прошел visible example и hidden generalization check. В текущем worker кандидаты "
                "выполняются последовательно, но логически это parallel-candidate selection."
            ),
        }
    return {"enabled": False}


def _foofah_memory_enabled(user_instruction: str | None) -> bool:
    for item in _json_objects_from_text(user_instruction):
        context = item.get("benchmark_context")
        if isinstance(context, dict) and "memory_enabled" in context:
            return bool(context.get("memory_enabled"))
        mode = item.get("foofah_memory_mode")
        if isinstance(mode, str):
            return mode.lower() in {"enabled", "true", "yes", "1"}
        if isinstance(mode, bool):
            return mode
    return True


def _normalize_foofah_router_plan(
    router_plan: dict | None,
    candidate_context: dict | None = None,
    failure_context: dict | None = None,
) -> dict:
    if not isinstance(router_plan, dict):
        return {"operator": "unknown", "confidence": 0.0, "parameters": {}, "ranked_operators": []}

    plan = dict(router_plan)
    ranked = [item for item in plan.get("ranked_operators", []) if isinstance(item, dict) and item.get("operator")]
    candidate = candidate_context or {}
    failure = failure_context or {}
    selected: dict | None = None

    if ranked and candidate.get("enabled"):
        index = max(1, int(candidate.get("candidate_index") or 1)) - 1
        selected = ranked[min(index, len(ranked) - 1)]
    elif ranked and failure.get("should_reroute"):
        failed_operator = failure.get("failed_operator")
        same_allowed = bool(plan.get("same_operator_allowed"))
        selected = next((item for item in ranked if same_allowed or item.get("operator") != failed_operator), ranked[0])
    elif ranked and not plan.get("operator"):
        selected = ranked[0]

    if failure.get("should_reroute") and ranked:
        failed_operator = failure.get("failed_operator")
        same_allowed = bool(plan.get("same_operator_allowed"))
        if (selected or plan).get("operator") == failed_operator and not same_allowed:
            alternative = next((item for item in ranked if item.get("operator") != failed_operator), None)
            if alternative:
                selected = alternative

    if selected:
        plan["operator"] = selected.get("operator", plan.get("operator", "unknown"))
        plan["confidence"] = selected.get("confidence", plan.get("confidence", 0.0))
        plan["parameters"] = selected.get("parameters", plan.get("parameters", {}))
        plan["generalization_assumption"] = selected.get(
            "generalization_assumption",
            plan.get("generalization_assumption", ""),
        )
        plan["candidate_selected_rank"] = ranked.index(selected) + 1 if selected in ranked else None
        plan["candidate_diversity_role"] = selected.get("diversity_role")
    if not plan.get("operator"):
        plan["operator"] = "unknown"
    plan["ranked_operators"] = ranked
    return plan


def _save_metric(
    db: Session,
    job: TransformJob,
    success: bool,
    latency_seconds: float,
    rows_processed: int = 0,
    columns_processed: int = 0,
    error_type: str | None = None,
    llm_calls: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
):
    db.add(
        JobMetric(
            job_id=job.id,
            success=success,
            attempts=job.attempts,
            latency_seconds=latency_seconds,
            rows_processed=rows_processed,
            columns_processed=columns_processed,
            error_type=error_type,
            llm_calls=llm_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            model_name=settings.effective_llm_model,
            cache_hit=False,
        )
    )
    db.commit()


def _attempt_history(db: Session, job_id: int) -> list[dict]:
    attempts = db.scalars(
        select(TransformAttempt).where(TransformAttempt.job_id == job_id).order_by(TransformAttempt.attempt_number)
    ).all()
    return [
        {
            "attempt": a.attempt_number,
            "success": a.success,
            "error": a.error_message,
            "explanation": a.explanation,
            "code_hash": _short_hash(a.generated_code),
            "plan_signature": None,
        }
        for a in attempts
    ]


def _lens_history(db: Session, job_id: int) -> list[dict]:
    from app.models.ala_lens import AlaLensEvent

    events = db.scalars(
        select(AlaLensEvent).where(AlaLensEvent.job_id == job_id).order_by(AlaLensEvent.id)
    ).all()
    return [
        {
            "attempt": e.attempt_number,
            "type": e.event_type,
            "note": e.note,
            "delta": e.delta_json,
            "amendment": e.amendment_json,
        }
        for e in events
    ]


def _record_attempt(
    db: Session,
    job: TransformJob,
    attempt_number: int,
    llm_result: LLMResult,
    success: bool,
    error_message: str | None = None,
    diff_json: str | None = None,
):
    db.add(
        TransformAttempt(
            job_id=job.id,
            attempt_number=attempt_number,
            phase="query" if job.mode == "query" else "transform",
            prompt_strategy=job.prompt_strategy or "standard",
            success=success,
            generated_code=llm_result.code,
            explanation=llm_result.explanation,
            error_message=error_message,
            diff_json=diff_json,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            total_tokens=llm_result.total_tokens,
            estimated_cost_usd=llm_result.estimated_cost_usd,
            latency_seconds=llm_result.latency_seconds,
        )
    )
    db.commit()


def _similar_transformations(db: Session, signature: str, limit: int = 3, prompt_version: str | None = None) -> list[dict]:
    if prompt_version is None:
        prompt_version = prompt_cache_version("foofah", cache_mode="honest") if signature.startswith("foofah:") else PROMPT_VERSION
    memories = db.scalars(
        select(TransformationMemory)
        .where(TransformationMemory.source_columns_signature == signature)
        .where(TransformationMemory.success.is_(True))
        .where(TransformationMemory.prompt_version == prompt_version)
        .order_by(TransformationMemory.uses.desc(), TransformationMemory.created_at.desc())
        .limit(limit)
    ).all()
    for memory in memories:
        memory.uses += 1
    db.commit()
    return [
        {
            "instruction": m.instruction,
            "code": m.generated_code,
            "explanation": m.explanation,
        }
        for m in memories
    ]


def _store_successful_memory(
    db: Session,
    source_df: pd.DataFrame,
    job: TransformJob,
    profile: dict,
    allow_foofah: bool = False,
) -> None:
    if not job.generated_code:
        return
    if job.prompt_strategy == "foofah" and not allow_foofah:
        return
    db.add(
        TransformationMemory(
            source_columns_signature=_memory_signature(source_df, job.prompt_strategy),
            instruction=job.user_instruction,
            generated_code=job.generated_code,
            explanation=job.explanation,
            profile_json=profile_to_json(profile),
            prompt_version=prompt_cache_version(job.prompt_strategy, job.user_instruction),
            success=True,
        )
    )
    db.commit()


def _mark_failed(
    db: Session,
    job: TransformJob,
    started: float,
    error: str,
    error_type: str,
    totals: dict,
) -> TransformJob:
    job.status = JobStatus.failed
    job.error_message = error
    job.updated_at = datetime.utcnow()
    db.commit()
    _save_metric(
        db,
        job,
        False,
        time.perf_counter() - started,
        error_type=error_type,
        llm_calls=totals["llm_calls"],
        prompt_tokens=totals["prompt_tokens"],
        completion_tokens=totals["completion_tokens"],
        total_tokens=totals["total_tokens"],
        estimated_cost_usd=totals["estimated_cost_usd"],
    )
    return job


def _add_usage(totals: dict, result: LLMResult) -> None:
    totals["llm_calls"] += 1
    totals["prompt_tokens"] += result.prompt_tokens
    totals["completion_tokens"] += result.completion_tokens
    totals["total_tokens"] += result.total_tokens
    totals["estimated_cost_usd"] += result.estimated_cost_usd


def _add_json_usage(totals: dict, result: JSONResult) -> None:
    totals["llm_calls"] += 1
    totals["prompt_tokens"] += result.prompt_tokens
    totals["completion_tokens"] += result.completion_tokens
    totals["total_tokens"] += result.total_tokens
    totals["estimated_cost_usd"] += result.estimated_cost_usd


def _classify_attempt_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "timeout" in message or "timed out" in message:
        return "sandbox_timeout"
    return type(exc).__name__


def _foofah_failure_context(
    last_error: str,
    diff_json: str | None,
    attempt_history: list[dict],
    router_plan: dict | None,
    current_code_hash: str | None = None,
    current_plan_signature: str | None = None,
) -> dict:
    details = {}
    if diff_json:
        try:
            details = json.loads(diff_json)
        except json.JSONDecodeError:
            details = {}
    column_diff = (details.get("column_diffs") or [{}])[0] if isinstance(details.get("column_diffs"), list) and details.get("column_diffs") else {}
    row_count = details.get("row_count") or {}
    value_diffs = details.get("value_diffs") or []
    rule_violations = details.get("rule_violations") or []

    actual_columns = column_diff.get("actual") or []
    expected_columns = column_diff.get("expected") or []
    actual_width = len(actual_columns) if actual_columns else None
    expected_width = len(expected_columns) if expected_columns else None
    actual_height = row_count.get("actual")
    expected_height = row_count.get("expected")
    value_diffs_by_row: dict[int, dict[str, dict]] = {}
    for diff in value_diffs:
        if not isinstance(diff, dict):
            continue
        row_index = diff.get("row")
        column = diff.get("column")
        if row_index is None or not column:
            continue
        value_diffs_by_row.setdefault(int(row_index), {})[str(column)] = diff
    leading_column_swap = any(
        row_diffs.get("col_0")
        and row_diffs.get("col_1")
        and str(row_diffs["col_0"].get("actual")) == str(row_diffs["col_1"].get("expected"))
        and str(row_diffs["col_1"].get("actual")) == str(row_diffs["col_0"].get("expected"))
        for row_diffs in value_diffs_by_row.values()
    )
    reordered_value_rows = 0
    err_like_value_mismatch = False
    for row_diffs in value_diffs_by_row.values():
        actuals = [str(item.get("actual")) for item in row_diffs.values()]
        expecteds = [str(item.get("expected")) for item in row_diffs.values()]
        if len(row_diffs) >= 4 and len(set(actuals) & set(expecteds)) >= 3:
            reordered_value_rows += 1
        if any(value.strip().lower().startswith("err:") for value in actuals + expecteds):
            err_like_value_mismatch = True

    error_family = "unknown"
    suggestions: list[str] = []
    if expected_width is not None and actual_width is not None:
        if actual_width < expected_width:
            error_family = "column_count_too_low"
            suggestions = ["wide_by_key", "pivot_long_to_wide", "context_attribute_pairs", "flatten_record_blocks", "prefix_suffix_chunks"]
        elif actual_width > expected_width:
            error_family = "column_count_too_high"
            suggestions = ["column_select", "split_extract expected width", "metadata/header handling", "fixed chunk width", "grouped_suffix_wide"]
        else:
            error_family = "column_order_mismatch"
            suggestions = ["preserve expected positional order", "rename/reorder output columns positionally"]
    elif expected_height is not None and actual_height is not None:
        if actual_height < expected_height:
            error_family = "row_count_too_low"
            suggestions = ["include row 0 if it is data", "scan all record blocks", "avoid over-filtering blanks"]
        elif actual_height > expected_height:
            error_family = "row_count_too_high"
            suggestions = ["skip metadata/header rows", "drop short title/metadata blocks", "do not emit attribute rows", "wrap_fixed_width", "merge repeated keys/blocks"]
        else:
            error_family = "row_count_mismatch"
    elif leading_column_swap:
        error_family = "leading_column_swap"
        suggestions = ["swap first two output fields", "header_grid_to_long positional order", "hidden feedback order override"]
    elif reordered_value_rows:
        error_family = "value_order_mismatch"
        suggestions = ["reorder output columns positionally", "stacked_metric_rows metric order", "wide_by_key", "prefix_suffix_chunks chunk order"]
    elif value_diffs:
        error_family = "value_mismatch"
        suggestions = ["preserve strings exactly", "check positional column order", "extract substrings without numeric parsing"]
        if err_like_value_mismatch:
            suggestions.extend(["prefix_suffix_chunks", "do not stop at first Err-like cell", "ignore only trailing all-error padding"])
    elif rule_violations:
        error_family = "rule_violation"
        suggestions = ["preserve string types", "avoid numeric/date conversion in FOOFAH"]
    elif last_error:
        error_family = "execution_or_static_error"
        suggestions = ["fix runtime/static validation error without changing inferred transformation"]

    failed_hashes = [item.get("code_hash") for item in attempt_history if item.get("code_hash")]
    failed_plan_signatures = [item.get("plan_signature") for item in attempt_history if item.get("plan_signature")]
    repeated_code_hash = bool(current_code_hash and current_code_hash in failed_hashes)
    repeated_plan_signature = bool(current_plan_signature and current_plan_signature in failed_plan_signatures)
    repeated_error_count = sum(1 for item in attempt_history if item.get("error") == last_error)
    anti_loop_requested = "anti-loop" in (last_error or "").lower()
    direct_repair_instruction = ""
    if leading_column_swap:
        direct_repair_instruction = (
            "Actual col_0 and col_1 are swapped against expected. Keep row generation and value extraction unchanged; "
            "swap only the first two emitted fields unless benchmark_generalization_analysis says visible and hidden require a shape-based branch."
        )
    elif reordered_value_rows:
        direct_repair_instruction = (
            "The same row values appear in actual and expected but in a different order. Keep extraction unchanged; build an explicit "
            "positional output order from expected cells. For metric pairs, test interleaved versus grouped metric order."
        )
    elif err_like_value_mismatch:
        direct_repair_instruction = (
            "Err-like cells appear in the mismatch. Do not stop scanning at the first Err-like cell; treat Err-like cells as padding "
            "only for trailing all-error chunks that the expected output omits."
        )

    return {
        "error_family": error_family,
        "message": last_error,
        "actual_width": actual_width,
        "expected_width": expected_width,
        "actual_height": actual_height,
        "expected_height": expected_height,
        "value_diff_count": len(value_diffs),
        "rule_violation_count": len(rule_violations),
        "operator_suggestions": suggestions,
        "leading_column_swap": leading_column_swap,
        "reordered_value_rows": reordered_value_rows,
        "err_like_value_mismatch": err_like_value_mismatch,
        "failed_operator": (router_plan or {}).get("operator"),
        "repeated_error_count": repeated_error_count,
        "repeated_code_hash": repeated_code_hash,
        "repeated_plan_signature": repeated_plan_signature,
        "direct_repair_instruction": direct_repair_instruction,
        "should_reroute": bool(anti_loop_requested or repeated_code_hash or repeated_plan_signature or repeated_error_count >= 2),
        "reroute_instruction": (
            "Choose a different operator family or materially different parameters when should_reroute is true; "
            "do not return the same operator with the same assumption after repeated failures."
        ),
    }


def _foofah_attempt_summary(
    attempt: int,
    repair_path: str,
    failure_context: dict | None,
    router_plan: dict | None,
    candidate_context: dict | None = None,
) -> dict:
    operator = (router_plan or {}).get("operator")
    candidate = candidate_context or {}
    candidate_prefix = ""
    if candidate.get("enabled"):
        candidate_prefix = f"candidate {candidate.get('candidate_index')}/{candidate.get('candidate_count')} in expensive multi-candidate mode; "
    if attempt == 1:
        return {
            "attempt": attempt,
            "repair_path": "initial_operator_generation",
            "summary": f"Attempt {attempt}: {candidate_prefix}initial FOOFAH operator routing selected {operator or 'unknown'} and generated the first candidate.",
            "reason": candidate.get("explanation") if candidate.get("enabled") else "No previous delta exists yet.",
            "operator": operator,
            "candidate_context": candidate if candidate.get("enabled") else None,
        }

    context = failure_context or {}
    error_family = context.get("error_family") or "unknown"
    suggestions = context.get("operator_suggestions") or []
    repeated_error_count = context.get("repeated_error_count") or 0
    repeated_code = bool(context.get("repeated_code_hash"))
    repeated_plan = bool(context.get("repeated_plan_signature"))

    if repair_path == "reroute":
        triggers = []
        if repeated_error_count >= 2:
            triggers.append(f"same error repeated {repeated_error_count} times")
        if repeated_code:
            triggers.append("candidate code repeated")
        if repeated_plan:
            triggers.append("candidate plan repeated")
        if not triggers:
            triggers.append("failure context requested operator reconsideration")
        reason = "; ".join(triggers)
        return {
            "attempt": attempt,
            "repair_path": "reroute",
            "summary": f"Attempt {attempt}: {candidate_prefix}reroute was used; the router reconsidered the operator after {error_family}.",
            "reason": reason,
            "operator": operator,
            "operator_suggestions": suggestions,
            "candidate_context": candidate if candidate.get("enabled") else None,
        }

    return {
        "attempt": attempt,
        "repair_path": "repair",
        "summary": f"Attempt {attempt}: {candidate_prefix}normal repair was used for {error_family}; the previous operator was kept and the code was amended.",
        "reason": "Reroute triggers were not reached yet.",
        "operator": operator,
        "operator_suggestions": suggestions,
        "candidate_context": candidate if candidate.get("enabled") else None,
    }


def _try_simple_query_result(source_df: pd.DataFrame, instruction: str | None) -> tuple[pd.DataFrame, str, str, dict] | None:
    query = (instruction or "").strip().lower()
    normalized = re.sub(r"\s+", " ", query)

    def result(df: pd.DataFrame, code: str, explanation: str, operation: str, parameters: dict | None = None):
        return df, code.strip() + "\n", explanation, {"operations": [operation], "parameters": parameters or {}, "deterministic_query": True}

    if not normalized or normalized in {
        "show table",
        "show all",
        "show data",
        "return table",
        "return all",
        "all rows",
        "display table",
        "показать таблицу",
        "покажи таблицу",
        "вывести таблицу",
        "вернуть таблицу",
        "все строки",
        "вся таблица",
    }:
        return result(
            source_df.copy(),
            "def transform(df):\n    return df.copy()",
            "Simple query fast path: returned the source table unchanged.",
            "return_source_table",
        )

    head_match = re.search(r"(?:first|head|первые|показать первые|вывести первые)\s+(\d+)", normalized)
    if head_match:
        n = max(1, int(head_match.group(1)))
        return result(
            source_df.head(n).copy(),
            f"def transform(df):\n    return df.head({n}).copy()",
            f"Simple query fast path: returned the first {n} rows.",
            "head",
            {"n": n},
        )

    tail_match = re.search(r"(?:last|tail|последние|показать последние|вывести последние)\s+(\d+)", normalized)
    if tail_match:
        n = max(1, int(tail_match.group(1)))
        return result(
            source_df.tail(n).copy(),
            f"def transform(df):\n    return df.tail({n}).copy()",
            f"Simple query fast path: returned the last {n} rows.",
            "tail",
            {"n": n},
        )

    if normalized in {"count rows", "row count", "rows count", "сколько строк", "количество строк", "посчитать строки"}:
        output = pd.DataFrame([{"row_count": len(source_df)}])
        return result(
            output,
            "def transform(df):\n    return pd.DataFrame([{'row_count': len(df)}])",
            "Simple query fast path: counted rows.",
            "count_rows",
        )

    if normalized in {"columns", "show columns", "list columns", "колонки", "список колонок", "названия колонок"}:
        output = pd.DataFrame({"column": list(source_df.columns)})
        return result(
            output,
            "def transform(df):\n    return pd.DataFrame({'column': list(df.columns)})",
            "Simple query fast path: listed columns.",
            "list_columns",
        )

    if normalized in {"describe", "summary", "statistics", "статистика", "описание", "сводка"}:
        output = source_df.describe(include="all").reset_index().rename(columns={"index": "statistic"})
        return result(
            output,
            "def transform(df):\n    return df.describe(include='all').reset_index().rename(columns={'index': 'statistic'})",
            "Simple query fast path: returned pandas describe(include='all').",
            "describe",
        )

    return None


def run_transform_job(db: Session, job: TransformJob, max_attempts_override: int | None = None) -> TransformJob:
    started = time.perf_counter()
    totals = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
    is_foofah = job.prompt_strategy == "foofah"
    candidate_context = _foofah_candidate_context(job.user_instruction) if is_foofah else {"enabled": False}
    max_attempts = max_attempts_override or (settings.foofah_max_repair_attempts if job.prompt_strategy == "foofah" else settings.max_repair_attempts)
    if not job.source_hash:
        fill_job_cache_keys(job)
    job.status = JobStatus.running
    job.updated_at = datetime.utcnow()
    db.commit()

    with trace_job(job.id, "tabular-transform-job", metadata={"mode": job.mode, "source_filename": job.source_filename}) as trace:
        try:
            with trace_span(trace, "read-and-profile-source", metadata={"source_path": job.source_path}):
                source_df = read_table(job.source_path, headerless=is_foofah)
                source_profile = dataframe_profile(source_df, max_rows=settings.max_prompt_example_rows)
            source_lens_model = source_model(source_df, source_profile)
            job.source_profile_json = profile_to_json(source_profile)
            db.commit()
            update_trace(trace, metadata={"source_rows": len(source_df), "source_columns": len(source_df.columns), "profile_warnings": source_profile.get("warnings", [])})

            if job.mode == "query":
                simple_query = _try_simple_query_result(source_df, job.user_instruction)
                if simple_query is not None:
                    actual, code, explanation, plan = simple_query
                    result_path = _save_result(job.id, actual)
                    job.attempts = 1
                    job.generated_code = code
                    job.explanation = explanation
                    job.result_path = result_path
                    job.status = JobStatus.success
                    job.error_message = None
                    job.updated_at = datetime.utcnow()
                    db.commit()
                    _record_attempt(
                        db,
                        job,
                        1,
                        LLMResult(code=code, explanation=explanation, plan=plan, raw_content=json.dumps({"code": code, "explanation": explanation, "plan": plan})),
                        True,
                    )
                    score_trace(trace, "job_success", 1.0, "Simple analytical query was handled without LLM.")
                    update_trace(trace, metadata={"status": "success", "attempts": 1, "total_tokens": 0}, output={"result_path": result_path})
                    _save_metric(
                        db,
                        job,
                        True,
                        time.perf_counter() - started,
                        len(actual),
                        len(actual.columns),
                        llm_calls=0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        estimated_cost_usd=0.0,
                    )
                    return job

            signature = _memory_signature(source_df, job.prompt_strategy)
            if is_foofah and not _foofah_memory_enabled(job.user_instruction):
                similar_examples = []
            else:
                similar_examples = _similar_transformations(
                    db,
                    signature,
                    prompt_version=prompt_cache_version(job.prompt_strategy, job.user_instruction),
                )
            sample_df = representative_sample(source_df, max_rows=settings.max_prompt_example_rows)
            with trace_span(trace, "read-expected-or-sample", metadata={"has_expected": bool(job.expected_path)}):
                expected_df = read_table(job.expected_path, headerless=is_foofah) if job.expected_path else None

            if job.mode == "query":
                with trace_span(trace, "build-query-prompt", metadata={"similar_examples": len(similar_examples)}):
                    prompt = build_query_prompt(
                        source_profile,
                        sample_df.to_dict(orient="records"),
                        job.user_instruction or "",
                        similar_examples,
                    )
            else:
                if expected_df is None:
                    raise ValueError("Expected file is required for transform mode.")
                validation_rows = len(source_df) if is_foofah else min(len(expected_df), len(source_df))
                example_source_df = source_df if is_foofah else source_df.head(validation_rows)
                example_input = example_source_df.to_dict(orient="records")
                example_output = expected_df.to_dict(orient="records")
                with trace_span(trace, "build-transform-prompt", metadata={"validation_rows": validation_rows, "similar_examples": len(similar_examples)}):
                    if is_foofah:
                        router_prompt = build_foofah_router_prompt(source_profile, example_input, example_output, job.user_instruction, similar_examples)
                        prompt = build_foofah_generation_prompt(source_profile, example_input, example_output, job.user_instruction, similar_examples)
                    else:
                        router_prompt = None
                        prompt = build_generation_prompt(source_profile, example_input, example_output, job.user_instruction, similar_examples)

            previous_code = ""
            last_error = ""
            last_error_type = "repair_failed"
            last_diff_json = None
            last_router_plan = None
            local_failure_history: list[dict] = []
            failure_context: dict | None = None
            llm = get_llm_client()

            for attempt in range(1, max_attempts + 1):
                job.attempts = attempt
                attempt_repair_path = "initial_operator_generation" if is_foofah else "initial"
                attempt_summary: dict | None = None
                if attempt == 1:
                    if is_foofah and router_prompt is not None:
                        with trace_span(trace, "llm-router", metadata={"attempt": attempt}):
                            router_result = llm.generate_json(
                                router_prompt,
                                trace=trace,
                                generation_name="foofah-operator-router",
                                system_prompt=FOOFAH_ROUTER_SYSTEM_PROMPT,
                        )
                        _add_json_usage(totals, router_result)
                        last_router_plan = _normalize_foofah_router_plan(router_result.data, candidate_context)
                        operator_prompt = build_foofah_operator_prompt(
                            last_router_plan,
                            source_profile,
                            example_input,
                            example_output,
                            job.user_instruction,
                            similar_examples,
                            failure_context=None,
                        )
                        with trace_span(trace, "llm-attempt", metadata={"attempt": attempt, "kind": "operator-code", "operator": last_router_plan.get("operator")}):
                            llm_result = llm.generate_code(
                                operator_prompt,
                                trace=trace,
                                generation_name="foofah-operator-code-generation",
                                system_prompt=FOOFAH_CODE_SYSTEM_PROMPT,
                            )
                    else:
                        with trace_span(trace, "llm-attempt", metadata={"attempt": attempt, "kind": "initial"}):
                            llm_result = llm.generate_code(
                                prompt,
                                trace=trace,
                                generation_name="initial-code-generation",
                                system_prompt=GENERIC_TRANSFORM_SYSTEM_PROMPT,
                            )
                else:
                    example_input = sample_df.to_dict(orient="records")
                    example_output = expected_df.to_dict(orient="records") if expected_df is not None else []
                    if is_foofah:
                        example_input = source_df.to_dict(orient="records")
                    if is_foofah:
                        failure_context = _foofah_failure_context(
                            last_error,
                            last_diff_json,
                            local_failure_history,
                            last_router_plan,
                        )
                    if is_foofah and failure_context and failure_context.get("should_reroute") and attempt >= 3:
                        attempt_repair_path = "reroute"
                        reroute_prompt = build_foofah_router_prompt(
                            source_profile,
                            example_input,
                            example_output,
                            job.user_instruction,
                            similar_examples,
                            reroute_context=failure_context,
                        )
                        with trace_span(trace, "llm-router-reroute", metadata={"attempt": attempt, "last_error": last_error, "failure_context": failure_context}):
                            router_result = llm.generate_json(
                                reroute_prompt,
                                trace=trace,
                                generation_name=f"foofah-reroute-attempt-{attempt}",
                                system_prompt=FOOFAH_ROUTER_SYSTEM_PROMPT,
                        )
                        _add_json_usage(totals, router_result)
                        last_router_plan = _normalize_foofah_router_plan(router_result.data, candidate_context, failure_context)
                        operator_prompt = build_foofah_operator_prompt(
                            last_router_plan,
                            source_profile,
                            example_input,
                            example_output,
                            job.user_instruction,
                            similar_examples,
                            failure_context=failure_context,
                        )
                        with trace_span(trace, "llm-attempt", metadata={"attempt": attempt, "kind": "rerouted-operator-code", "operator": last_router_plan.get("operator")}):
                            llm_result = llm.generate_code(
                                operator_prompt,
                                trace=trace,
                                generation_name=f"foofah-rerouted-code-attempt-{attempt}",
                                system_prompt=FOOFAH_CODE_SYSTEM_PROMPT,
                            )
                    else:
                        attempt_repair_path = "repair" if is_foofah else "repair"
                        lens_putback_context = None
                        if is_foofah and last_diff_json:
                            try:
                                last_delta_payload = json.loads(last_diff_json)
                            except json.JSONDecodeError:
                                last_delta_payload = {}
                            normalized_delta = typed_view_delta(last_delta_payload, failure_context)
                            normalized_policy = putback_policy(normalized_delta, last_router_plan, failure_context)
                            lens_putback_context = {
                                "typed_view_delta": normalized_delta,
                                "putback_policy": normalized_policy,
                                "putback_mode": putback_mode(normalized_policy, {"parameter_putback_supported": True, "source_putback_supported": False}),
                                "source_mutation_allowed": False,
                                "putback_target": normalized_policy.get("putback_target"),
                                "instruction": (
                                    "Use this normalized lens putback context as the repair target. "
                                    "Source mutation is forbidden; amend only transformation parameter p / generated Python."
                                ),
                            }
                        repair_prompt = build_repair_prompt(
                            previous_code,
                            last_error,
                            source_profile,
                            example_input,
                            example_output,
                            job.user_instruction,
                            _attempt_history(db, job.id) + ([{"router_plan": last_router_plan}] if last_router_plan else []),
                            mode="foofah_program_synthesis_repair" if is_foofah else job.mode,
                            failure_context=failure_context if is_foofah else None,
                            lens_putback_context=lens_putback_context,
                        )
                        with trace_span(trace, "llm-attempt", metadata={"attempt": attempt, "kind": "repair", "last_error": last_error, "failure_context": failure_context or {}}):
                            llm_result = llm.generate_code(
                                repair_prompt,
                                trace=trace,
                                generation_name=f"repair-attempt-{attempt}",
                                system_prompt=FOOFAH_CODE_SYSTEM_PROMPT if is_foofah else GENERIC_TRANSFORM_SYSTEM_PROMPT,
                            )

                previous_parameter = parameter_model(previous_code, None, None, event_type="amendment") if previous_code else None
                _add_usage(totals, llm_result)
                if is_foofah and last_router_plan:
                    llm_result.plan = {
                        **(llm_result.plan or {}),
                        "router_operator": last_router_plan.get("operator"),
                        "router_confidence": last_router_plan.get("confidence"),
                        "router_parameters": last_router_plan.get("parameters", {}),
                        "router_generalization_assumption": last_router_plan.get("generalization_assumption", ""),
                    }
                if is_foofah:
                    attempt_summary = _foofah_attempt_summary(attempt, attempt_repair_path, failure_context, last_router_plan, candidate_context)
                current_code_hash = _short_hash(llm_result.code)
                current_plan_signature = _plan_signature(llm_result.plan)
                repeated_failed_candidate = bool(
                    is_foofah
                    and any(
                        (current_code_hash and item.get("code_hash") == current_code_hash)
                        or (current_plan_signature and item.get("plan_signature") == current_plan_signature)
                        for item in local_failure_history
                    )
                )
                job.generated_code = llm_result.code
                job.explanation = llm_result.explanation
                db.commit()
                current_parameter = parameter_model(
                    llm_result.code,
                    llm_result.explanation,
                    llm_result.plan,
                    attempt_history=local_failure_history,
                    failure_context=failure_context,
                    event_type="get" if attempt == 1 else "amendment",
                )
                if candidate_context.get("enabled"):
                    current_parameter["candidate_context"] = candidate_context
                if attempt == 1:
                    record_lens_event(
                        db,
                        job.id,
                        attempt,
                        "get",
                        prompt_strategy=job.prompt_strategy,
                        source=source_lens_model,
                        view=view_model(expected_df) if expected_df is not None else None,
                        parameter_after=current_parameter,
                        note=(attempt_summary or {}).get("summary") or "Initial parameter p was synthesized from source model and expected view example.",
                    )
                else:
                    record_lens_event(
                        db,
                        job.id,
                        attempt,
                        "amendment",
                        prompt_strategy=job.prompt_strategy,
                        source=source_lens_model,
                        view=view_model(expected_df) if expected_df is not None else None,
                        parameter_before=previous_parameter,
                        amendment={
                            "repair_prompt_error": last_error,
                            "repair_path": attempt_repair_path,
                            "repair_path_reason": (attempt_summary or {}).get("reason"),
                            "attempt_summary": attempt_summary,
                            "candidate_context": candidate_context if candidate_context.get("enabled") else None,
                            "failure_context": failure_context,
                            "attempt_history": _attempt_history(db, job.id),
                        },
                        parameter_after=current_parameter,
                        note=(attempt_summary or {}).get("summary") or "Parameter p was amended after a validation delta.",
                    )
                previous_code = llm_result.code

                if repeated_failed_candidate and attempt < max_attempts:
                    last_error = "Anti-loop: generated code or plan repeated a previous failed FOOFAH candidate; reroute required."
                    last_error_type = "anti_loop_repeated_candidate"
                    local_failure_history.append(
                        {
                            "attempt": attempt,
                            "error": last_error,
                            "code_hash": current_code_hash,
                            "plan_signature": current_plan_signature,
                            "operator": (last_router_plan or {}).get("operator"),
                        }
                    )
                    _record_attempt(db, job, attempt, llm_result, False, last_error, last_diff_json)
                    continue

                if settings.max_total_llm_tokens_per_job and totals["total_tokens"] > settings.max_total_llm_tokens_per_job:
                    last_error = "LLM token budget exceeded."
                    last_error_type = "token_budget_exceeded"
                    _record_attempt(db, job, attempt, llm_result, False, last_error, last_diff_json)
                    local_failure_history.append(
                        {
                            "attempt": attempt,
                            "error": last_error,
                            "code_hash": current_code_hash,
                            "plan_signature": current_plan_signature,
                            "operator": (last_router_plan or {}).get("operator"),
                        }
                    )
                    break

                try:
                    with trace_span(trace, "static-validation", metadata={"attempt": attempt}):
                        validate_code_safety(llm_result.code)
                        if is_foofah:
                            validate_foofah_matrix_style(llm_result.code)
                    if job.mode == "query":
                        with trace_span(trace, "sandbox-query-execution", metadata={"attempt": attempt, "rows": len(source_df)}):
                            actual = execute_code_in_sandbox(llm_result.code, source_df)
                        if not isinstance(actual, pd.DataFrame) or actual.empty:
                            raise ValueError("Analytical query returned an empty or invalid DataFrame.")
                        result_path = _save_result(job.id, actual)
                        job.status = JobStatus.success
                        job.result_path = result_path
                        job.error_message = None
                        job.updated_at = datetime.utcnow()
                        db.commit()
                        _record_attempt(db, job, attempt, llm_result, True)
                        score_trace(trace, "job_success", 1.0, "Analytical query returned a non-empty DataFrame.")
                        update_trace(trace, metadata={"status": "success", "attempts": attempt, "total_tokens": totals["total_tokens"]}, output={"result_path": result_path})
                        _store_successful_memory(db, source_df, job, source_profile)
                        _save_metric(
                            db,
                            job,
                            True,
                            time.perf_counter() - started,
                            len(actual),
                            len(actual.columns),
                            llm_calls=totals["llm_calls"],
                            prompt_tokens=totals["prompt_tokens"],
                            completion_tokens=totals["completion_tokens"],
                            total_tokens=totals["total_tokens"],
                            estimated_cost_usd=totals["estimated_cost_usd"],
                        )
                        return job

                    assert expected_df is not None
                    validation_df = source_df if is_foofah else source_df.head(min(len(expected_df), len(source_df)))
                    with trace_span(trace, "sandbox-validation-execution", metadata={"attempt": attempt, "validation_rows": len(validation_df)}):
                        actual = execute_code_in_sandbox(llm_result.code, validation_df, string_mode=is_foofah)
                    with trace_span(trace, "compare-actual-expected", metadata={"attempt": attempt}):
                        if is_foofah:
                            actual = _as_positional_table(actual)
                            expected_for_compare = _as_positional_table(expected_df)
                        else:
                            expected_for_compare = expected_df
                        report = compare_dataframes_report(
                            actual,
                            expected_for_compare,
                            business_rules=None if is_foofah else infer_rules_from_expected(expected_for_compare),
                            exact_strings=is_foofah,
                        )
                    if (
                        not report.ok
                        and not is_foofah
                        and len(validation_df) < len(source_df)
                        and (report.details.get("row_count") or {}).get("actual", 0) < (report.details.get("row_count") or {}).get("expected", 0)
                    ):
                        with trace_span(
                            trace,
                            "sandbox-full-source-validation-retry",
                            metadata={
                                "attempt": attempt,
                                "validation_rows": len(validation_df),
                                "full_rows": len(source_df),
                                "reason": "row_count_too_low_after_head_validation",
                            },
                        ):
                            full_source_actual = execute_code_in_sandbox(llm_result.code, source_df, string_mode=False)
                        with trace_span(trace, "compare-full-source-actual-expected", metadata={"attempt": attempt}):
                            full_source_report = compare_dataframes_report(
                                full_source_actual,
                                expected_for_compare,
                                business_rules=infer_rules_from_expected(expected_for_compare),
                            )
                        if full_source_report.ok:
                            actual = full_source_actual
                            report = full_source_report
                    job.validation_report_json = report.to_json()
                    db.commit()
                    if not report.ok:
                        last_error = report.message
                        last_error_type = "validation_failure"
                        last_diff_json = report.to_json()
                        local_failure_history.append(
                            {
                                "attempt": attempt,
                                "error": last_error,
                                "code_hash": current_code_hash,
                                "plan_signature": current_plan_signature,
                                "operator": (last_router_plan or {}).get("operator"),
                                "diff_json": last_diff_json,
                            }
                        )
                        delta_failure_context = failure_context
                        if is_foofah:
                            delta_failure_context = _foofah_failure_context(
                                last_error,
                                last_diff_json,
                                local_failure_history,
                                last_router_plan,
                                current_code_hash,
                                current_plan_signature,
                            )
                        score_trace(trace, f"attempt_{attempt}_match", 0.0, report.message)
                        delta_payload = json.loads(last_diff_json)
                        delta_parameter = parameter_model(
                            llm_result.code,
                            llm_result.explanation,
                            llm_result.plan,
                            validation_report=delta_payload,
                            attempt_history=local_failure_history,
                            failure_context=delta_failure_context,
                            event_type="delta",
                        )
                        if candidate_context.get("enabled"):
                            delta_parameter["candidate_context"] = candidate_context
                        record_lens_event(
                            db,
                            job.id,
                            attempt,
                            "delta",
                            prompt_strategy=job.prompt_strategy,
                            source=source_lens_model,
                            view=view_model(expected_df),
                            parameter_before=delta_parameter,
                            delta=delta_payload,
                            note="Actual view did not match expected view; repair loop will amend p.",
                        )
                        _record_attempt(db, job, attempt, llm_result, False, last_error, last_diff_json)
                        continue

                    with trace_span(trace, "sandbox-full-execution", metadata={"attempt": attempt, "rows": len(source_df)}):
                        final_df = execute_code_in_sandbox(llm_result.code, source_df, string_mode=is_foofah)
                    if is_foofah:
                        final_df = _as_positional_table(final_df)
                    result_path = _save_result(job.id, final_df)
                    job.status = JobStatus.success
                    job.result_path = result_path
                    job.error_message = None
                    job.updated_at = datetime.utcnow()
                    db.commit()
                    _record_attempt(db, job, attempt, llm_result, True, diff_json=report.to_json())
                    score_trace(trace, f"attempt_{attempt}_match", 1.0, "Expected example matched.")
                    score_trace(trace, "job_success", 1.0, "Transform job succeeded.")
                    update_trace(trace, metadata={"status": "success", "attempts": attempt, "total_tokens": totals["total_tokens"], "estimated_cost_usd": totals["estimated_cost_usd"]}, output={"result_path": result_path})
                    stability_payload = json.loads(report.to_json())
                    stable_parameter = parameter_model(
                        llm_result.code,
                        llm_result.explanation,
                        llm_result.plan,
                        validation_report=stability_payload,
                        attempt_history=local_failure_history,
                        failure_context=failure_context,
                        event_type="stability",
                    )
                    if candidate_context.get("enabled"):
                        stable_parameter["candidate_context"] = candidate_context
                    record_lens_event(
                        db,
                        job.id,
                        attempt,
                        "stability",
                        prompt_strategy=job.prompt_strategy,
                        source=source_lens_model,
                        view=view_model(final_df.head(settings.max_prompt_example_rows)),
                        parameter_after=stable_parameter,
                        note="The amended parameter p produced a view consistent with the expected example.",
                    )
                    _store_successful_memory(db, source_df, job, source_profile)
                    _save_metric(
                        db,
                        job,
                        True,
                        time.perf_counter() - started,
                        len(final_df),
                        len(final_df.columns),
                        llm_calls=totals["llm_calls"],
                        prompt_tokens=totals["prompt_tokens"],
                        completion_tokens=totals["completion_tokens"],
                        total_tokens=totals["total_tokens"],
                        estimated_cost_usd=totals["estimated_cost_usd"],
                    )
                    return job
                except Exception as exc:
                    last_error = str(exc)
                    last_error_type = _classify_attempt_error(exc)
                    local_failure_history.append(
                        {
                            "attempt": attempt,
                            "error": last_error,
                            "code_hash": current_code_hash,
                            "plan_signature": current_plan_signature,
                            "operator": (last_router_plan or {}).get("operator"),
                            "diff_json": last_diff_json,
                        }
                    )
                    score_trace(trace, f"attempt_{attempt}_error", 0.0, last_error)
                    _record_attempt(db, job, attempt, llm_result, False, last_error, last_diff_json)
                    continue

            score_trace(trace, "job_success", 0.0, last_error)
            update_trace(trace, metadata={"status": "failed", "attempts": job.attempts, "total_tokens": totals["total_tokens"]}, output={"error": last_error})
            return _mark_failed(db, job, started, last_error, last_error_type, totals)
        except Exception as exc:
            score_trace(trace, "job_success", 0.0, str(exc))
            update_trace(trace, metadata={"status": "failed", "total_tokens": totals["total_tokens"]}, output={"error": str(exc)})
            return _mark_failed(db, job, started, str(exc), type(exc).__name__, totals)
