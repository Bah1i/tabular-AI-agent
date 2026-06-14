import time
import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.benchmark import BenchmarkCaseResult
from app.models.job import JobStatus, TransformJob
from app.models.metric import JobMetric
from app.services.prompts import PROMPT_VERSION
from app.utils.hashing import compute_file_hash, compute_many_text_hash


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


def foofah_cache_mode_from_instruction(user_instruction: str | None) -> str:
    for item in _json_objects_from_text(user_instruction):
        context = item.get("benchmark_context")
        if isinstance(context, dict):
            mode = context.get("benchmark_mode")
            if isinstance(mode, str) and mode.lower() in {"strict_honest", "middle_honest", "oracle"}:
                return mode.lower()
        mode = item.get("foofah_cache_mode") or item.get("foofah_benchmark_mode")
        if isinstance(mode, str) and mode.lower() in {"oracle", "dishonest", "hidden_oracle"}:
            return "oracle"
        if isinstance(mode, str) and mode.lower() in {"middle_honest", "little_tricky"}:
            return "middle_honest"
    return "strict_honest"


def prompt_cache_version(prompt_strategy: str | None, user_instruction: str | None = None, cache_mode: str | None = None) -> str:
    if prompt_strategy != "foofah":
        return PROMPT_VERSION
    mode = (cache_mode or foofah_cache_mode_from_instruction(user_instruction)).lower()
    if mode in {"dishonest", "hidden_oracle"}:
        mode = "oracle"
    elif mode in {"honest", "clean", "strict"}:
        mode = "strict_honest"
    elif mode in {"little_tricky", "middle"}:
        mode = "middle_honest"
    elif mode not in {"strict_honest", "middle_honest", "oracle"}:
        mode = "strict_honest"
    return f"{PROMPT_VERSION}|{mode}"


def fill_job_cache_keys(job: TransformJob) -> None:
    job.source_hash = compute_file_hash(job.source_path)
    job.expected_hash = compute_file_hash(job.expected_path) if job.expected_path else None
    job.instruction_hash = compute_many_text_hash(job.user_instruction, job.prompt_strategy)
    job.model_name = settings.effective_llm_model
    job.prompt_version = prompt_cache_version(job.prompt_strategy, job.user_instruction)


def find_successful_cache_source(db: Session, job: TransformJob) -> TransformJob | None:
    if not job.source_hash or not job.instruction_hash or not job.model_name or not job.prompt_version:
        return None
    statement = (
        select(TransformJob)
        .where(TransformJob.id != job.id)
        .where(TransformJob.status == JobStatus.success)
        .where(TransformJob.mode == job.mode)
        .where(TransformJob.source_hash == job.source_hash)
        .where(TransformJob.expected_hash == job.expected_hash)
        .where(TransformJob.instruction_hash == job.instruction_hash)
        .where(TransformJob.model_name == job.model_name)
        .where(TransformJob.prompt_version == job.prompt_version)
        .where(TransformJob.result_path.is_not(None))
    )
    if job.prompt_strategy == "foofah":
        statement = statement.join(BenchmarkCaseResult, BenchmarkCaseResult.job_id == TransformJob.id).where(
            BenchmarkCaseResult.success.is_(True)
        )
    statement = statement.order_by(TransformJob.updated_at.desc(), TransformJob.id.desc()).limit(1)
    return db.scalars(statement).first()


def apply_cache_hit_if_available(db: Session, job: TransformJob) -> bool:
    if job.status == JobStatus.success and job.cache_hit_from_job_id:
        return True
    started = time.perf_counter()
    cached = find_successful_cache_source(db, job)
    if not cached:
        return False

    job.status = JobStatus.success
    job.cache_hit_from_job_id = cached.id
    job.result_path = cached.result_path
    job.generated_code = cached.generated_code
    job.explanation = cached.explanation
    job.source_profile_json = cached.source_profile_json
    job.validation_report_json = cached.validation_report_json
    job.error_message = None
    job.attempts = 0
    job.updated_at = datetime.utcnow()
    db.add(
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
    db.commit()
    return True
