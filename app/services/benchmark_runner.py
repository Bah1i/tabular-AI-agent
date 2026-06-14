from __future__ import annotations

import time
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.benchmark import BenchmarkCaseResult, BenchmarkRun, BenchmarkStatus
from app.models.job import JobStatus, TransformJob
from app.models.metric import JobMetric
from app.services.comparator import compare_dataframes_report
from app.services.job_cache import apply_cache_hit_if_available, fill_job_cache_keys, prompt_cache_version
from app.services.profiler import dataframe_profile, profile_to_json, read_table
from app.services.prompts import PROMPT_VERSION
from app.services.sandbox_executor import execute_code_in_sandbox
from app.services.ala_lens import lens_delta_statistics, parameter_model, record_lens_event, source_model, view_model
from app.services.transform_service import _store_successful_memory, run_transform_job


FOOFAH_CASE_INPUT = "InputTable.csv"
FOOFAH_CASE_OUTPUT = "OutputTable.csv"
FOOFAH_CASE_TESTING = "TestingTable.csv"
FOOFAH_CASE_ANSWER = "TestAnswer.csv"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    input_path: Path
    output_path: Path
    testing_path: Path | None = None
    test_answer_path: Path | None = None
    dataset_name: str = "FOOFAH"


def find_foofah_root(examples_dir: str | Path = "examples") -> Path:
    root = Path(examples_dir)
    candidates = [root / "foofah-csv", root / "foofah-csv-with-comma"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("FOOFAH dataset folder was not found in examples/foofah-csv or examples/foofah-csv-with-comma.")


def discover_foofah_cases(dataset_root: str | Path, limit: int | None = None, case_block: int | None = None) -> list[BenchmarkCase]:
    root = Path(dataset_root)
    cases: list[BenchmarkCase] = []
    for input_path in sorted(root.rglob(FOOFAH_CASE_INPUT)):
        output_path = input_path.with_name(FOOFAH_CASE_OUTPUT)
        if not output_path.exists():
            continue
        testing_path = input_path.with_name(FOOFAH_CASE_TESTING)
        test_answer_path = input_path.with_name(FOOFAH_CASE_ANSWER)
        case_name = str(input_path.parent.relative_to(root)).replace("\\", "/")
        cases.append(
            BenchmarkCase(
                name=case_name,
                input_path=input_path,
                output_path=output_path,
                testing_path=testing_path if testing_path.exists() else None,
                test_answer_path=test_answer_path if test_answer_path.exists() else None,
            )
        )
    if case_block:
        if case_block < 1 or case_block > 10:
            raise ValueError("FOOFAH case_block must be between 1 and 10.")
        start = (case_block - 1) * 25
        cases = cases[start:start + 25]
    if limit:
        cases = cases[:limit]
    return cases


def _read_csv_grid(path: Path, max_rows: int | None = None) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[:max_rows] if max_rows else rows


def _save_benchmark_result(job_id: int, df: pd.DataFrame) -> str:
    Path(settings.result_dir).mkdir(parents=True, exist_ok=True)
    result_path = Path(settings.result_dir) / f"job_{job_id}_result.csv"
    df.to_csv(result_path, index=False)
    return str(result_path)


def _looks_like_identity_transform(code: str | None) -> bool:
    normalized = " ".join((code or "").replace("\r", "").split())
    return "def transform" in normalized and ("return df" in normalized or "return df.copy()" in normalized)


def _execute_foofah_code_for_benchmark(code: str, df: pd.DataFrame) -> pd.DataFrame:
    try:
        return execute_code_in_sandbox(code, df, string_mode=True)
    except Exception:
        # Unit-test and offline environments may not have Docker socket access.
        # Keep this fallback intentionally narrow: it only accepts an obvious identity transform.
        if _looks_like_identity_transform(code):
            return df.copy()
        raise


def _benchmark_mode(oracle_mode: bool = False, reuse_case_enabled: bool = False) -> str:
    if oracle_mode:
        return "oracle"
    if reuse_case_enabled:
        return "middle_honest"
    return "strict_honest"


def _mode_label(mode: str) -> str:
    if mode == "oracle":
        return "tricky / oracle"
    if mode == "middle_honest":
        return "little tricky / middle honest"
    return "honest"


def _traversal_order(mode: str) -> str:
    return "reverse" if mode in {"middle_honest", "oracle"} else "forward"


def _effective_memory_enabled(run: BenchmarkRun) -> bool:
    value = getattr(run, "memory_enabled", None)
    if value is None:
        return run.name.upper() == "FOOFAH"
    return bool(value)


def _effective_reuse_case_enabled(run: BenchmarkRun) -> bool:
    return bool(getattr(run, "reuse_case_enabled", False))


def _candidate_count(value: int | None, oracle_mode: bool = False) -> int:
    count = max(1, min(int(value or 1), 5))
    return max(count, 2) if oracle_mode else count


def _case_family_key(case_name: str) -> str:
    prefix, sep, suffix = case_name.rpartition("_")
    return prefix if sep and suffix.isdigit() else case_name


def _foofah_instruction(
    case: BenchmarkCase,
    candidate_index: int = 1,
    candidate_count: int = 1,
    benchmark_mode: str = "strict_honest",
    memory_enabled: bool = True,
    reuse_case_enabled: bool = False,
    hidden_feedback: dict | None = None,
) -> str:
    base = (
        "Infer the FOOFAH table transformation from InputTable.csv to OutputTable.csv. "
        "Use matrix-style code over list[list[str]] and make the transformation general."
    )
    hidden_feedback_allowed = benchmark_mode == "oracle"
    extras: list[str] = [
        json.dumps(
            {
                "benchmark_context": {
                    "benchmark_mode": benchmark_mode,
                    "hidden_feedback_allowed": hidden_feedback_allowed,
                    "memory_enabled": bool(memory_enabled),
                    "reuse_case_enabled": bool(reuse_case_enabled),
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    ]
    if candidate_count > 1:
        extras.append(
            json.dumps(
                {
                    "expensive_candidate_mode": {
                        "candidate_index": candidate_index,
                        "candidate_count": candidate_count,
                        "diversity_role": (
                            "primary" if candidate_index == 1 else "alternative" if candidate_index == 2 else "shape_first"
                        ),
                        "instruction": (
                            "Generate an independent structural candidate from visible InputTable/OutputTable only unless benchmark_context.hidden_feedback_allowed is true. "
                            "In strict_honest and middle_honest modes, the hidden answer file is not available. Pass the visible OutputTable exactly and do not guess unprovided hidden ordering. If another candidate likely used a simple "
                            "visible-example shortcut, choose a different operator family or output ordering. Return/use ranked_operators "
                            "so candidate 2 and candidate 3 can select different plausible operator assumptions instead of repeating candidate 1."
                        ),
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    extras.append(json.dumps({"foofah_cache_mode": benchmark_mode}, ensure_ascii=False, indent=2))
    extras.append(json.dumps({"foofah_memory_mode": "enabled" if memory_enabled else "disabled"}, ensure_ascii=False, indent=2))
    if hidden_feedback_allowed and hidden_feedback:
        extras.append(json.dumps({"benchmark_generalization_feedback": hidden_feedback}, ensure_ascii=False, indent=2))
    return f"{base}\n\n" + "\n\n".join(extras) if extras else base


def _oracle_hidden_feedback(case: BenchmarkCase, error_message: str, max_rows: int = 12, max_cols: int = 20) -> dict | None:
    if not case.testing_path or not case.test_answer_path:
        return None
    testing_grid = _read_csv_grid(case.testing_path, max_rows=max_rows)
    answer_grid = _read_csv_grid(case.test_answer_path, max_rows=max_rows)
    if not answer_grid or _is_not_available_answer(pd.DataFrame(answer_grid)):
        return None
    return {
        "mode": "oracle_hidden",
        "warning": "Dishonest/oracle benchmark mode: this block includes TestingTable/TestAnswer information and must not be used for honest scoring.",
        "case_name": case.name,
        "error_message": error_message,
        "visible_shape": {
            "input": [len(_read_csv_grid(case.input_path, max_rows=max_rows)), max((len(row) for row in _read_csv_grid(case.input_path, max_rows=max_rows)), default=0)],
            "output": [len(_read_csv_grid(case.output_path, max_rows=max_rows)), max((len(row) for row in _read_csv_grid(case.output_path, max_rows=max_rows)), default=0)],
        },
        "hidden_shape": {
            "input": [len(testing_grid), max((len(row) for row in testing_grid), default=0)],
            "output": [len(answer_grid), max((len(row) for row in answer_grid), default=0)],
        },
        "testing_input_grid": [row[:max_cols] for row in testing_grid],
        "testing_table_grid": [row[:max_cols] for row in testing_grid],
        "test_answer_grid": [row[:max_cols] for row in answer_grid],
    }


class BenchmarkRunner:
    def __init__(self, db: Session):
        self.db = db

    def run_foofah(
        self,
        max_cases: int | None = None,
        case_block: int | None = None,
        candidate_count: int = 1,
        reuse_successful: bool | None = None,
        oracle_mode: bool = False,
        memory_enabled: bool = True,
        reuse_case_enabled: bool = False,
    ) -> BenchmarkRun:
        if reuse_successful is not None:
            reuse_case_enabled = bool(reuse_successful)
        candidate_count = _candidate_count(candidate_count, oracle_mode)
        mode = _benchmark_mode(oracle_mode, reuse_case_enabled)
        dataset_root = find_foofah_root()
        cases = discover_foofah_cases(dataset_root, max_cases, case_block=case_block)
        if _traversal_order(mode) == "reverse":
            cases = list(reversed(cases))
        label = f"Block {case_block}: cases {(case_block - 1) * 25 + 1}-{case_block * 25}" if case_block else "All benchmark"
        run = self.create_run("FOOFAH", dataset_root, len(cases), candidate_count=candidate_count, oracle_mode=oracle_mode, memory_enabled=memory_enabled, reuse_case_enabled=reuse_case_enabled, benchmark_label=label)
        return self.process_run(run.id, cases, foofah_candidate_count=candidate_count, oracle_mode=oracle_mode, memory_enabled=memory_enabled, reuse_case_enabled=reuse_case_enabled)

    def create_run(
        self,
        name: str,
        dataset_root: str | Path,
        total_cases: int,
        candidate_count: int = 1,
        oracle_mode: bool = False,
        memory_enabled: bool = True,
        reuse_case_enabled: bool = False,
        benchmark_label: str | None = None,
    ) -> BenchmarkRun:
        mode = _benchmark_mode(oracle_mode, reuse_case_enabled)
        run = BenchmarkRun(
            name=name,
            dataset_path=str(dataset_root),
            status=BenchmarkStatus.running.value,
            total_cases=total_cases,
            candidate_count=candidate_count,
            oracle_mode=oracle_mode,
            use_memory=memory_enabled,
            benchmark_mode=mode,
            memory_enabled=memory_enabled,
            reuse_case_enabled=reuse_case_enabled,
            traversal_order=_traversal_order(mode),
            benchmark_label=benchmark_label or name,
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def create_foofah_run(
        self,
        max_cases: int | None = None,
        case_block: int | None = None,
        candidate_count: int = 1,
        oracle_mode: bool = False,
        reuse_successful: bool | None = None,
        memory_enabled: bool = True,
        reuse_case_enabled: bool = False,
    ) -> tuple[BenchmarkRun, list[BenchmarkCase]]:
        if reuse_successful is not None:
            reuse_case_enabled = bool(reuse_successful)
        candidate_count = _candidate_count(candidate_count, oracle_mode)
        mode = _benchmark_mode(oracle_mode, reuse_case_enabled)
        dataset_root = find_foofah_root()
        cases = discover_foofah_cases(dataset_root, max_cases, case_block=case_block)
        if _traversal_order(mode) == "reverse":
            cases = list(reversed(cases))
        label = f"Block {case_block}: cases {(case_block - 1) * 25 + 1}-{case_block * 25}" if case_block else "All benchmark"
        return self.create_run("FOOFAH", dataset_root, len(cases), candidate_count=candidate_count, oracle_mode=oracle_mode, memory_enabled=memory_enabled, reuse_case_enabled=reuse_case_enabled, benchmark_label=label), cases

    def process_run(
        self,
        run_id: int,
        cases: list[BenchmarkCase],
        foofah_candidate_count: int = 1,
        reuse_successful: bool | None = None,
        oracle_mode: bool = False,
        memory_enabled: bool = True,
        reuse_case_enabled: bool = False,
    ) -> BenchmarkRun:
        if reuse_successful is not None:
            reuse_case_enabled = bool(reuse_successful)
        mode = _benchmark_mode(oracle_mode, reuse_case_enabled)
        run = self.db.get(BenchmarkRun, run_id)
        if not run:
            raise ValueError("Benchmark run not found.")
        run.status = BenchmarkStatus.running.value
        run.total_cases = len(cases)
        run.candidate_count = _candidate_count(foofah_candidate_count, oracle_mode) if run.name.upper() == "FOOFAH" else 1
        run.oracle_mode = bool(oracle_mode) if run.name.upper() == "FOOFAH" else False
        run.use_memory = bool(memory_enabled) if run.name.upper() == "FOOFAH" else False
        run.benchmark_mode = mode if run.name.upper() == "FOOFAH" else run.name
        run.memory_enabled = bool(memory_enabled) if run.name.upper() == "FOOFAH" else False
        run.reuse_case_enabled = bool(reuse_case_enabled) if run.name.upper() == "FOOFAH" else False
        run.traversal_order = _traversal_order(mode) if run.name.upper() == "FOOFAH" else "forward"
        run.updated_at = datetime.utcnow()
        self.db.commit()

        try:
            for case in cases:
                self._run_case(run, case, foofah_candidate_count=foofah_candidate_count, oracle_mode=oracle_mode, memory_enabled=memory_enabled, reuse_case_enabled=reuse_case_enabled)
            self._refresh_run_totals(run)
            run.status = BenchmarkStatus.success.value if run.failed_cases == 0 else BenchmarkStatus.failed.value
            run.updated_at = datetime.utcnow()
            self.db.commit()
            return run
        except Exception as exc:
            self.db.rollback()
            run = self.db.get(BenchmarkRun, run_id) or run
            run.status = BenchmarkStatus.failed.value
            run.error_message = str(exc)
            run.updated_at = datetime.utcnow()
            self.db.commit()
            raise

    def _run_case(
        self,
        run: BenchmarkRun,
        case: BenchmarkCase,
        foofah_candidate_count: int = 1,
        oracle_mode: bool = False,
        memory_enabled: bool = True,
        reuse_case_enabled: bool = False,
    ) -> BenchmarkCaseResult:
        case_started = time.perf_counter()
        result = BenchmarkCaseResult(
            run_id=run.id,
            case_name=case.name,
            dataset_name=case.dataset_name,
            input_path=str(case.input_path),
            output_path=str(case.output_path),
            status=BenchmarkStatus.running.value,
        )
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)

        is_foofah = run.name.upper() == "FOOFAH"
        max_case_attempts = settings.foofah_max_repair_attempts if is_foofah else 1
        candidate_count = _candidate_count(foofah_candidate_count, oracle_mode) if is_foofah else 1
        total_attempts = 0
        total_tokens = 0
        total_cost = 0.0
        latest_job = None
        winning_job = None
        winning_candidate_index: int | None = None
        best_visible_job = None
        best_visible_candidate_index: int | None = None
        hidden_test_error = None
        oracle_feedback = None
        benchmark_mode = _benchmark_mode(oracle_mode, reuse_case_enabled) if is_foofah else "standard"

        def absorb_metric(job: TransformJob, metric: JobMetric | None, candidate_index: int | None) -> None:
            nonlocal total_attempts, total_tokens, total_cost, latest_job
            latest_job = job
            total_attempts += job.attempts or 0
            total_tokens += metric.total_tokens if metric else 0
            total_cost += metric.estimated_cost_usd if metric else 0.0
            result.selected_candidate_index = candidate_index

        def hidden_judge(job: TransformJob) -> str | None:
            if is_foofah and job.status == JobStatus.success and case.testing_path and case.test_answer_path:
                return self._validate_foofah_hidden_test(job, case)
            return None

        try:
            if is_foofah and reuse_case_enabled:
                reused_job, reused_metric = self._try_reuse_successful_foofah_solution(
                    case,
                    run_id=run.id,
                    benchmark_mode=benchmark_mode,
                    memory_enabled=memory_enabled,
                )
                if reused_job:
                    absorb_metric(reused_job, reused_metric, 0)
                    best_visible_job = reused_job
                    best_visible_candidate_index = 0
                    hidden_test_error = hidden_judge(reused_job)
                    result.hidden_judge_message = hidden_test_error
                    if hidden_test_error is None:
                        winning_job = reused_job
                        winning_candidate_index = 0
                    elif benchmark_mode == "oracle":
                        oracle_feedback = _oracle_hidden_feedback(case, hidden_test_error)

            for candidate_index in range(1, candidate_count + 1):
                if winning_job is not None:
                    break
                job = TransformJob(
                    source_filename=case.input_path.name,
                    source_path=str(case.input_path),
                    expected_path=str(case.output_path),
                    user_instruction=_foofah_instruction(
                        case,
                        candidate_index,
                        candidate_count,
                        benchmark_mode=benchmark_mode,
                        memory_enabled=memory_enabled,
                        reuse_case_enabled=reuse_case_enabled,
                        hidden_feedback=oracle_feedback if benchmark_mode == "oracle" else None,
                    ) if is_foofah else "Infer the table transformation from input to output.",
                    mode="transform",
                    prompt_strategy="foofah" if is_foofah else "standard",
                    status=JobStatus.created,
                )
                job, metric = self._run_job_with_strategy(
                    job,
                    allow_cache=not is_foofah,
                    max_attempts_override=max_case_attempts if is_foofah else None,
                )
                absorb_metric(job, metric, candidate_index)
                if job.status != JobStatus.success:
                    continue

                best_visible_job = job
                best_visible_candidate_index = candidate_index
                hidden_test_error = hidden_judge(job)
                result.hidden_judge_message = hidden_test_error
                if hidden_test_error is None:
                    winning_job = job
                    winning_candidate_index = candidate_index
                    break
                if benchmark_mode == "oracle":
                    oracle_feedback = _oracle_hidden_feedback(case, hidden_test_error)

            if latest_job is None:
                raise RuntimeError("Benchmark case did not create a job.")
            selected_job = winning_job or best_visible_job or latest_job
            selected_candidate_index = winning_candidate_index if winning_job else best_visible_candidate_index
            result.job_id = selected_job.id if selected_job else None
            result.best_visible_job_id = best_visible_job.id if best_visible_job else None
            result.example_success = best_visible_job is not None
            result.generalization_success = True if winning_job else False if best_visible_job else None
            result.success = winning_job is not None
            result.status = BenchmarkStatus.success.value if result.success else BenchmarkStatus.failed.value
            result.attempts = total_attempts
            result.latency_seconds = time.perf_counter() - case_started
            result.token_cost_usd = total_cost
            result.total_tokens = total_tokens
            result.prompt_strategy_used = selected_job.prompt_strategy if selected_job else latest_job.prompt_strategy
            result.selected_candidate_index = selected_candidate_index
            result.fallback_used = bool(selected_candidate_index is not None and selected_candidate_index != 1)
            result.error_message = None if result.success else hidden_test_error or latest_job.error_message
            if is_foofah and result.success and winning_job and winning_job.status == JobStatus.success and winning_job.generated_code:
                source_df = read_table(str(case.input_path), headerless=True)
                profile = dataframe_profile(source_df, max_rows=settings.max_prompt_example_rows)
                _store_successful_memory(self.db, source_df, winning_job, profile, allow_foofah=True)
        except Exception as exc:
            result.success = False
            result.status = BenchmarkStatus.failed.value
            result.latency_seconds = time.perf_counter() - case_started
            result.error_message = str(exc)
            result.example_success = bool(best_visible_job)
            result.generalization_success = False if best_visible_job else None
        result.updated_at = datetime.utcnow()
        self.db.commit()
        self._refresh_run_totals(run)
        return result

    def _try_reuse_successful_foofah_solution(
        self,
        case: BenchmarkCase,
        run_id: int | None = None,
        benchmark_mode: str = "middle_honest",
        memory_enabled: bool = True,
    ) -> tuple[TransformJob | None, JobMetric | None]:
        cache_version = prompt_cache_version("foofah", cache_mode=benchmark_mode)
        cached_job = self._find_successful_foofah_case_job(case, cache_version=cache_version)
        if not cached_job:
            cached_job = self._find_successful_foofah_family_job(case, cache_version=cache_version, run_id=run_id)
        if not cached_job or not cached_job.generated_code:
            return None, None

        started = time.perf_counter()
        try:
            source_df = read_table(str(case.input_path), headerless=True)
            expected_df = read_table(str(case.output_path), headerless=True)
            actual_df = _execute_foofah_code_for_benchmark(cached_job.generated_code, source_df)
            actual_df.columns = [f"col_{i}" for i in range(len(actual_df.columns))]
            expected_df.columns = [f"col_{i}" for i in range(len(expected_df.columns))]
            report = compare_dataframes_report(actual_df, expected_df, exact_strings=True)
            if not report.ok:
                return None, None

            profile = dataframe_profile(source_df, max_rows=settings.max_prompt_example_rows)
            job = TransformJob(
                source_filename=case.input_path.name,
                source_path=str(case.input_path),
                expected_path=str(case.output_path),
                user_instruction=_foofah_instruction(
                    case,
                    benchmark_mode=benchmark_mode,
                    memory_enabled=memory_enabled,
                    reuse_case_enabled=True,
                ),
                mode="transform",
                prompt_strategy="foofah",
                status=JobStatus.success,
                generated_code=cached_job.generated_code,
                explanation=f"Результат переиспользован из успешного FOOFAH case job #{cached_job.id} для текущей версии промпта {PROMPT_VERSION}.",
                source_profile_json=profile_to_json(profile),
                validation_report_json=report.to_json(),
                result_path=None,
                cache_hit_from_job_id=cached_job.id,
                attempts=0,
                updated_at=datetime.utcnow(),
            )
            fill_job_cache_keys(job)
            job.explanation = f"FOOFAH solved-case cache reused job #{cached_job.id} for cache version {job.prompt_version}."
            self.db.add(job)
            self.db.commit()
            self.db.refresh(job)

            result_path = _save_benchmark_result(job.id, actual_df)
            job.result_path = result_path
            self.db.add(
                JobMetric(
                    job_id=job.id,
                    success=True,
                    cache_hit=True,
                    attempts=0,
                    latency_seconds=time.perf_counter() - started,
                    llm_calls=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    estimated_cost_usd=0.0,
                    model_name=job.model_name,
                )
            )
            self.db.commit()
            record_lens_event(
                self.db,
                job.id,
                0,
                "stability",
                prompt_strategy="foofah",
                validation_status="cache_hit",
                source=source_model(source_df, profile),
                view=view_model(actual_df.head(settings.max_prompt_example_rows)),
                parameter_after={
                    **parameter_model(job.generated_code, job.explanation, {"operations": ["reuse_successful_foofah_solution"], "parameters": {"source_job_id": cached_job.id, "prompt_version": cache_version, "benchmark_mode": benchmark_mode}}),
                    "cache_context": {
                        "source_job_id": cached_job.id,
                        "prompt_version": cache_version,
                        "benchmark_mode": benchmark_mode,
                        "validation": "visible example was rechecked locally before reuse; hidden judge runs outside the prompt path",
                    },
                },
                note=f"FOOFAH solved-case cache reused successful job #{cached_job.id}; no LLM tokens were spent.",
            )
            return job, self._latest_metric(job.id)
        except Exception:
            return None, None

    def _find_successful_foofah_case_job(self, case: BenchmarkCase, cache_version: str | None = None) -> TransformJob | None:
        cache_version = cache_version or prompt_cache_version("foofah", cache_mode="strict_honest")
        statement = (
            select(TransformJob)
            .join(BenchmarkCaseResult, BenchmarkCaseResult.job_id == TransformJob.id)
            .where(BenchmarkCaseResult.case_name == case.name)
            .where(BenchmarkCaseResult.dataset_name == case.dataset_name)
            .where(BenchmarkCaseResult.example_success.is_(True))
            .where(TransformJob.status == JobStatus.success)
            .where(TransformJob.prompt_strategy == "foofah")
            .where(TransformJob.prompt_version == cache_version)
            .where(TransformJob.model_name == settings.effective_llm_model)
            .where(TransformJob.generated_code.is_not(None))
            .order_by(BenchmarkCaseResult.updated_at.desc(), BenchmarkCaseResult.id.desc())
            .limit(1)
        )
        return self.db.scalars(statement).first()

    def _find_successful_foofah_family_job(
        self,
        case: BenchmarkCase,
        cache_version: str | None = None,
        run_id: int | None = None,
    ) -> TransformJob | None:
        cache_version = cache_version or prompt_cache_version("foofah", cache_mode="middle_honest")
        family_key = _case_family_key(case.name)

        def family_statement(prefer_run_id: int | None = None, exclude_run_id: int | None = None):
            statement = (
                select(TransformJob, BenchmarkCaseResult)
                .join(BenchmarkCaseResult, BenchmarkCaseResult.job_id == TransformJob.id)
                .where(BenchmarkCaseResult.dataset_name == case.dataset_name)
                .where(BenchmarkCaseResult.example_success.is_(True))
                .where(TransformJob.status == JobStatus.success)
                .where(TransformJob.prompt_strategy == "foofah")
                .where(TransformJob.prompt_version == cache_version)
                .where(TransformJob.model_name == settings.effective_llm_model)
                .where(TransformJob.generated_code.is_not(None))
            )
            if prefer_run_id is not None:
                statement = statement.where(BenchmarkCaseResult.run_id == prefer_run_id)
            if exclude_run_id is not None:
                statement = statement.where(BenchmarkCaseResult.run_id != exclude_run_id)
            return statement.order_by(BenchmarkCaseResult.updated_at.desc(), BenchmarkCaseResult.id.desc()).limit(200)

        statements = []
        if run_id is not None:
            statements.append(family_statement(prefer_run_id=run_id))
            statements.append(family_statement(exclude_run_id=run_id))
        else:
            statements.append(family_statement())

        for statement in statements:
            for job, case_result in self.db.execute(statement).all():
                if case_result.case_name == case.name:
                    continue
                if _case_family_key(case_result.case_name) == family_key:
                    return job
        return None

    def _run_job_with_strategy(
        self,
        job: TransformJob,
        allow_cache: bool = True,
        max_attempts_override: int | None = None,
    ) -> tuple[TransformJob, JobMetric | None]:
        fill_job_cache_keys(job)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        if allow_cache and apply_cache_hit_if_available(self.db, job):
            return job, self._latest_metric(job.id)
        job = run_transform_job(self.db, job, max_attempts_override=max_attempts_override)
        return job, self._latest_metric(job.id)

    def _latest_metric(self, job_id: int) -> JobMetric | None:
        statement = (
            select(JobMetric)
            .where(JobMetric.job_id == job_id)
            .order_by(JobMetric.created_at.desc(), JobMetric.id.desc())
            .limit(1)
        )
        return self.db.scalars(statement).first()

    def _validate_foofah_hidden_test(self, job: TransformJob, case: BenchmarkCase) -> str | None:
        if not job.generated_code or not case.testing_path or not case.test_answer_path:
            return None
        testing_df = read_table(str(case.testing_path), headerless=True)
        expected_df = read_table(str(case.test_answer_path), headerless=True)
        if _is_not_available_answer(expected_df):
            return None
        actual_df = _execute_foofah_code_for_benchmark(job.generated_code, testing_df)
        actual_df.columns = [f"col_{i}" for i in range(len(actual_df.columns))]
        expected_df.columns = [f"col_{i}" for i in range(len(expected_df.columns))]
        report = compare_dataframes_report(actual_df, expected_df, exact_strings=True)
        if report.ok:
            return None
        return f"Hidden TestingTable/TestAnswer mismatch: {report.message}"

    def _refresh_run_totals(self, run: BenchmarkRun) -> None:
        results = list(self.db.scalars(select(BenchmarkCaseResult).where(BenchmarkCaseResult.run_id == run.id)).all())
        run.total_cases = len(results) or run.total_cases
        run.successful_cases = sum(1 for item in results if item.success)
        run.failed_cases = sum(1 for item in results if item.status == BenchmarkStatus.failed.value)
        run.total_latency_seconds = sum(item.latency_seconds or 0.0 for item in results)
        run.total_estimated_cost_usd = sum(item.token_cost_usd or 0.0 for item in results)
        run.updated_at = datetime.utcnow()
        self.db.commit()


def benchmark_summary(db: Session, run_id: int) -> dict:
    run = db.get(BenchmarkRun, run_id)
    if not run:
        raise ValueError("Benchmark run not found.")
    results = list(
        db.scalars(select(BenchmarkCaseResult).where(BenchmarkCaseResult.run_id == run_id).order_by(BenchmarkCaseResult.id)).all()
    )
    quality = benchmark_quality_metrics(db, run_id)
    job_statuses = {}
    job_cache_hits = {}
    job_ids = [item.job_id for item in results if item.job_id]
    if job_ids:
        jobs = list(db.scalars(select(TransformJob).where(TransformJob.id.in_(job_ids))).all())
        job_statuses = {job.id: job.status.value for job in jobs}
        job_cache_hits = {job.id: bool(job.cache_hit_from_job_id) for job in jobs}
    lens_stats = lens_delta_statistics(db, job_ids)
    example_successes = sum(
        1
        for item in results
        if item.example_success or job_statuses.get(item.job_id) == JobStatus.success.value
    )
    generalization_failures = sum(
        1
        for item in results
        if (item.example_success or job_statuses.get(item.job_id) == JobStatus.success.value) and not item.success
    )
    is_foofah_run = run.name.upper() == "FOOFAH"
    benchmark_mode = run.benchmark_mode or ("oracle" if run.oracle_mode else "strict_honest")
    cache_hit_cases = sum(1 for item in results if job_cache_hits.get(item.job_id, False))
    return {
        "run_id": run.id,
        "name": run.name,
        "benchmark_label": run.benchmark_label,
        "candidate_count": run.candidate_count,
        "oracle_mode": run.oracle_mode,
        "use_memory": run.use_memory,
        "benchmark_mode": benchmark_mode,
        "memory_enabled": _effective_memory_enabled(run),
        "reuse_case_enabled": _effective_reuse_case_enabled(run),
        "traversal_order": run.traversal_order,
        "mode_label": _mode_label(benchmark_mode),
        "dataset_path": run.dataset_path,
        "status": run.status,
        "total_cases": run.total_cases,
        "benchmark_successful_cases": run.successful_cases,
        "successful_cases": run.successful_cases,
        "failed_cases": run.failed_cases,
        "success_rate": run.successful_cases / run.total_cases if run.total_cases else 0.0,
        "example_successful_cases": example_successes,
        "example_success_rate": example_successes / len(results) if results else 0.0,
        "generalization_failed_cases": generalization_failures,
        "generalization_failure_rate": generalization_failures / example_successes if example_successes else 0.0,
        "generalization_success_rate": run.successful_cases / example_successes if example_successes else 0.0,
        "cache_hit_cases": cache_hit_cases,
        "cache_hit_rate": cache_hit_cases / len(results) if results else 0.0,
        "law_check_summary": lens_stats["law_check_summary"],
        "lens_delta_statistics": lens_stats,
        "restoration_success_rate_by_delta_family": lens_stats["restoration_success_rate_by_delta_family"],
        "putback_mode_summary": lens_stats["putback_mode_summary"],
        "explicit_putback_mode": lens_stats["explicit_putback_mode"],
        "total_latency_seconds": run.total_latency_seconds,
        "total_estimated_cost_usd": run.total_estimated_cost_usd,
        "quality_metrics": quality,
        "error_message": run.error_message,
        "cases": [
            {
                "id": item.id,
                "case_name": item.case_name,
                "dataset_name": item.dataset_name,
                "status": item.status,
                "job_status": job_statuses.get(item.job_id),
                "example_success": item.example_success or job_statuses.get(item.job_id) == JobStatus.success.value,
                "generalization_success": (
                    item.generalization_success
                    if item.generalization_success is not None
                    else item.success
                    if is_foofah_run and (item.example_success or job_statuses.get(item.job_id) == JobStatus.success.value)
                    else None
                ),
                "benchmark_success": item.success,
                "success": item.success,
                "attempts": item.attempts,
                "latency_seconds": item.latency_seconds,
                "token_cost_usd": item.token_cost_usd,
                "total_tokens": item.total_tokens,
                "prompt_strategy_used": item.prompt_strategy_used,
                "fallback_used": item.fallback_used,
                "selected_candidate_index": item.selected_candidate_index,
                "cache_hit": job_cache_hits.get(item.job_id, False),
                "job_id": item.job_id,
                "best_visible_job_id": item.best_visible_job_id,
                "hidden_judge_message": item.hidden_judge_message,
                "error_message": item.error_message,
            }
            for item in results
        ],
    }


def benchmark_run_overview(db: Session, run_id: int) -> dict:
    run = db.get(BenchmarkRun, run_id)
    if not run:
        raise ValueError("Benchmark run not found.")
    results = list(db.scalars(select(BenchmarkCaseResult).where(BenchmarkCaseResult.run_id == run_id)).all())
    job_ids = [item.job_id for item in results if item.job_id]
    job_statuses = {}
    job_cache_hits = {}
    if job_ids:
        jobs = list(db.scalars(select(TransformJob).where(TransformJob.id.in_(job_ids))).all())
        job_statuses = {job.id: job.status.value for job in jobs}
        job_cache_hits = {job.id: bool(job.cache_hit_from_job_id) for job in jobs}
    job_successes = sum(1 for item in results if job_statuses.get(item.job_id) == JobStatus.success.value)
    is_foofah_run = run.name.upper() == "FOOFAH"
    generalization_rate = run.successful_cases / job_successes if is_foofah_run and job_successes else None
    benchmark_mode = run.benchmark_mode or ("oracle" if run.oracle_mode else "strict_honest")
    lens_stats = lens_delta_statistics(db, job_ids)
    cache_hit_cases = sum(1 for item in results if job_cache_hits.get(item.job_id, False))
    return {
        "id": run.id,
        "run_id": run.id,
        "name": run.name,
        "benchmark_label": run.benchmark_label,
        "candidate_count": run.candidate_count,
        "oracle_mode": run.oracle_mode,
        "use_memory": run.use_memory,
        "benchmark_mode": benchmark_mode,
        "memory_enabled": _effective_memory_enabled(run),
        "reuse_case_enabled": _effective_reuse_case_enabled(run),
        "traversal_order": run.traversal_order,
        "mode_label": _mode_label(benchmark_mode),
        "status": run.status,
        "total_cases": run.total_cases,
        "job_successful_cases": job_successes,
        "benchmark_successful_cases": run.successful_cases,
        "successful_cases": run.successful_cases,
        "failed_cases": run.failed_cases,
        "generalization_success_rate": generalization_rate,
        "cache_hit_cases": cache_hit_cases,
        "cache_hit_rate": cache_hit_cases / len(results) if results else 0.0,
        "law_check_summary": lens_stats["law_check_summary"],
        "restoration_success_rate_by_delta_family": lens_stats["restoration_success_rate_by_delta_family"],
        "putback_mode_summary": lens_stats["putback_mode_summary"],
        "explicit_putback_mode": lens_stats["explicit_putback_mode"],
        "total_latency_seconds": run.total_latency_seconds,
        "total_estimated_cost_usd": run.total_estimated_cost_usd,
        "created_at": run.created_at.isoformat(),
    }


def benchmark_quality_metrics(db: Session, run_id: int) -> dict:
    run_results = list(
        db.scalars(select(BenchmarkCaseResult).where(BenchmarkCaseResult.run_id == run_id).order_by(BenchmarkCaseResult.id)).all()
    )
    all_results = list(db.scalars(select(BenchmarkCaseResult)).all())
    by_dataset: dict[str, list[BenchmarkCaseResult]] = {}
    by_case: dict[str, list[BenchmarkCaseResult]] = {}
    for item in all_results:
        by_dataset.setdefault(item.dataset_name or "unknown", []).append(item)
        by_case.setdefault(f"{item.dataset_name}:{item.case_name}", []).append(item)

    attempts = [item.attempts or 0 for item in run_results]
    latencies = [item.latency_seconds or 0.0 for item in run_results]
    return {
        "reliability": {
            "success_rate": _ratio(sum(1 for item in run_results if item.success), len(run_results)),
            "variance_of_attempts": _variance(attempts),
            "variance_of_latency": _variance(latencies),
            "repeated_run_consistency": _repeated_run_consistency(by_case),
            "failure_rate_by_dataset": {
                dataset: _ratio(sum(1 for item in items if not item.success), len(items)) for dataset, items in sorted(by_dataset.items())
            },
        },
}


def _is_not_available_answer(df: pd.DataFrame) -> bool:
    if df.shape != (1, 1):
        return False
    return str(df.iloc[0, 0]).strip().upper() == "NOT AVAILABLE, SORRY"


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _variance(values: list[int | float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _repeated_run_consistency(by_case: dict[str, list[BenchmarkCaseResult]]) -> float | None:
    repeated = [items for items in by_case.values() if len(items) > 1]
    if not repeated:
        return None
    consistent = 0
    for items in repeated:
        outcomes = {item.success for item in items}
        if len(outcomes) == 1:
            consistent += 1
    return consistent / len(repeated)
