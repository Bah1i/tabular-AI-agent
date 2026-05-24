import json
import pathlib
import time
from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.attempt import TransformAttempt
from app.models.job import TransformJob, JobStatus
from app.models.memory import TransformationMemory
from app.models.metric import JobMetric
from app.services.ala_lens import parameter_model, record_lens_event, source_model, view_model
from app.services.comparator import compare_dataframes_report, infer_rules_from_expected
from app.services.llm_client import DeepSeekClient, LLMResult
from app.services.profiler import (
    columns_signature,
    dataframe_profile,
    profile_to_json,
    read_table,
    representative_sample,
)
from app.services.prompts import build_generation_prompt, build_query_prompt, build_repair_prompt
from app.services.sandbox_executor import execute_code_in_sandbox
from app.services.static_validator import validate_code_safety
from app.services.tracing import trace_job


def _save_result(job_id: int, df: pd.DataFrame) -> str:
    pathlib.Path(settings.result_dir).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(settings.result_dir) / f"job_{job_id}_result.csv"
    df.to_csv(p, index=False)
    return str(p)


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
            model_name=settings.deepseek_model,
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


def _similar_transformations(db: Session, signature: str, limit: int = 3) -> list[dict]:
    memories = db.scalars(
        select(TransformationMemory)
        .where(TransformationMemory.source_columns_signature == signature)
        .where(TransformationMemory.success.is_(True))
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


def _store_successful_memory(db: Session, source_df: pd.DataFrame, job: TransformJob, profile: dict) -> None:
    if not job.generated_code:
        return
    db.add(
        TransformationMemory(
            source_columns_signature=columns_signature(source_df),
            instruction=job.user_instruction,
            generated_code=job.generated_code,
            explanation=job.explanation,
            profile_json=profile_to_json(profile),
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


def run_transform_job(db: Session, job: TransformJob) -> TransformJob:
    started = time.perf_counter()
    totals = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
    job.status = JobStatus.running
    job.updated_at = datetime.utcnow()
    db.commit()
    llm = DeepSeekClient()

    with trace_job(job.id, "tabular-transform-job") as trace:
        try:
            source_df = read_table(job.source_path)
            source_profile = dataframe_profile(source_df, max_rows=settings.max_prompt_example_rows)
            source_lens_model = source_model(source_df, source_profile)
            job.source_profile_json = profile_to_json(source_profile)
            db.commit()

            signature = columns_signature(source_df)
            similar_examples = _similar_transformations(db, signature)
            sample_df = representative_sample(source_df, max_rows=settings.max_prompt_example_rows)
            expected_df = read_table(job.expected_path) if job.expected_path else None

            if job.mode == "query":
                prompt = build_query_prompt(
                    source_profile,
                    sample_df.to_dict(orient="records"),
                    job.user_instruction or "",
                    similar_examples,
                )
            else:
                if expected_df is None:
                    raise ValueError("Expected file is required for transform mode.")
                validation_rows = min(len(expected_df), len(source_df))
                example_input = source_df.head(validation_rows).to_dict(orient="records")
                example_output = expected_df.to_dict(orient="records")
                prompt = build_generation_prompt(source_profile, example_input, example_output, job.user_instruction, similar_examples)

            previous_code = ""
            last_error = ""
            last_diff_json = None

            for attempt in range(1, settings.max_repair_attempts + 1):
                job.attempts = attempt
                if attempt == 1:
                    llm_result = llm.generate_code(prompt, trace=trace, generation_name="initial-code-generation")
                else:
                    example_input = sample_df.to_dict(orient="records")
                    example_output = expected_df.to_dict(orient="records") if expected_df is not None else []
                    repair_prompt = build_repair_prompt(
                        previous_code,
                        last_error,
                        source_profile,
                        example_input,
                        example_output,
                        job.user_instruction,
                        _attempt_history(db, job.id),
                        mode=job.mode,
                    )
                    llm_result = llm.generate_code(repair_prompt, trace=trace, generation_name=f"repair-attempt-{attempt}")

                previous_parameter = parameter_model(previous_code, None, None) if previous_code else None
                _add_usage(totals, llm_result)
                job.generated_code = llm_result.code
                job.explanation = llm_result.explanation
                db.commit()
                current_parameter = parameter_model(llm_result.code, llm_result.explanation, llm_result.plan)
                if attempt == 1:
                    record_lens_event(
                        db,
                        job.id,
                        attempt,
                        "get",
                        source=source_lens_model,
                        view=view_model(expected_df) if expected_df is not None else None,
                        parameter_after=current_parameter,
                        note="Initial parameter p was synthesized from source model and expected view example.",
                    )
                else:
                    record_lens_event(
                        db,
                        job.id,
                        attempt,
                        "amendment",
                        source=source_lens_model,
                        view=view_model(expected_df) if expected_df is not None else None,
                        parameter_before=previous_parameter,
                        amendment={"repair_prompt_error": last_error, "attempt_history": _attempt_history(db, job.id)},
                        parameter_after=current_parameter,
                        note="Parameter p was amended after a validation delta.",
                    )
                previous_code = llm_result.code

                if settings.max_total_llm_tokens_per_job and totals["total_tokens"] > settings.max_total_llm_tokens_per_job:
                    last_error = "LLM token budget exceeded."
                    _record_attempt(db, job, attempt, llm_result, False, last_error, last_diff_json)
                    break

                try:
                    validate_code_safety(llm_result.code)
                    if job.mode == "query":
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
                    validation_rows = min(len(expected_df), len(source_df))
                    actual = execute_code_in_sandbox(llm_result.code, source_df.head(validation_rows))
                    report = compare_dataframes_report(
                        actual,
                        expected_df,
                        business_rules=infer_rules_from_expected(expected_df),
                    )
                    job.validation_report_json = report.to_json()
                    db.commit()
                    if not report.ok:
                        last_error = report.message
                        last_diff_json = report.to_json()
                        record_lens_event(
                            db,
                            job.id,
                            attempt,
                            "delta",
                            source=source_lens_model,
                            view=view_model(expected_df),
                            parameter_before=current_parameter,
                            delta=json.loads(last_diff_json),
                            note="Actual view did not match expected view; repair loop will amend p.",
                        )
                        _record_attempt(db, job, attempt, llm_result, False, last_error, last_diff_json)
                        continue

                    final_df = execute_code_in_sandbox(llm_result.code, source_df)
                    result_path = _save_result(job.id, final_df)
                    job.status = JobStatus.success
                    job.result_path = result_path
                    job.error_message = None
                    job.updated_at = datetime.utcnow()
                    db.commit()
                    _record_attempt(db, job, attempt, llm_result, True, diff_json=report.to_json())
                    record_lens_event(
                        db,
                        job.id,
                        attempt,
                        "stability",
                        source=source_lens_model,
                        view=view_model(final_df.head(settings.max_prompt_example_rows)),
                        parameter_after=current_parameter,
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
                    _record_attempt(db, job, attempt, llm_result, False, last_error, last_diff_json)
                    continue

            return _mark_failed(db, job, started, last_error, "repair_failed", totals)
        except Exception as exc:
            return _mark_failed(db, job, started, str(exc), type(exc).__name__, totals)
