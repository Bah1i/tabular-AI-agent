import pathlib, time
from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.job import TransformJob, JobStatus
from app.models.metric import JobMetric
from app.services.comparator import compare_dataframes
from app.services.llm_client import DeepSeekClient
from app.services.profiler import read_table, dataframe_profile
from app.services.prompts import build_generation_prompt, build_repair_prompt
from app.services.sandbox_executor import execute_code_in_sandbox
from app.services.static_validator import validate_code_safety
from app.services.tracing import trace_job
def _save_result(job_id: int, df: pd.DataFrame) -> str:
    pathlib.Path(settings.result_dir).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(settings.result_dir) / f'job_{job_id}_result.csv'; df.to_csv(p, index=False); return str(p)
def _save_metric(db: Session, job: TransformJob, success: bool, latency_seconds: float, rows_processed: int = 0, columns_processed: int = 0, error_type: str | None = None):
    db.add(JobMetric(job_id=job.id, success=success, attempts=job.attempts, latency_seconds=latency_seconds, rows_processed=rows_processed, columns_processed=columns_processed, error_type=error_type)); db.commit()
def run_transform_job(db: Session, job: TransformJob) -> TransformJob:
    started = time.perf_counter(); job.status = JobStatus.running; job.updated_at = datetime.utcnow(); db.commit(); llm = DeepSeekClient()
    with trace_job(job.id, 'tabular-transform-job') as trace:
        try:
            source_df = read_table(job.source_path); expected_df = read_table(job.expected_path)
            profile = dataframe_profile(source_df); ex_in = source_df.head(len(expected_df)).to_dict(orient='records'); ex_out = expected_df.to_dict(orient='records')
            prompt = build_generation_prompt(profile, ex_in, ex_out, job.user_instruction)
            previous_code = ''; last_error = ''
            for attempt in range(1, settings.max_repair_attempts + 1):
                job.attempts = attempt
                if attempt == 1: code, _ = llm.generate_code(prompt, trace=trace, generation_name='initial-code-generation')
                else: code, _ = llm.generate_code(build_repair_prompt(previous_code, last_error, profile, ex_in, ex_out, job.user_instruction), trace=trace, generation_name=f'repair-attempt-{attempt}')
                previous_code = code; job.generated_code = code; db.commit()
                try:
                    validate_code_safety(code)
                    actual = execute_code_in_sandbox(code, source_df.head(len(expected_df)))
                    ok, diff = compare_dataframes(actual, expected_df)
                    if not ok: last_error = diff; continue
                    final_df = execute_code_in_sandbox(code, source_df); result_path = _save_result(job.id, final_df)
                    job.status = JobStatus.success; job.result_path = result_path; job.error_message = None; job.updated_at = datetime.utcnow(); db.commit()
                    _save_metric(db, job, True, time.perf_counter()-started, len(final_df), len(final_df.columns)); return job
                except Exception as exc:
                    last_error = str(exc); continue
            job.status = JobStatus.failed; job.error_message = last_error; job.updated_at = datetime.utcnow(); db.commit(); _save_metric(db, job, False, time.perf_counter()-started, error_type='repair_failed'); return job
        except Exception as exc:
            job.status = JobStatus.failed; job.error_message = str(exc); job.updated_at = datetime.utcnow(); db.commit(); _save_metric(db, job, False, time.perf_counter()-started, error_type=type(exc).__name__); return job
