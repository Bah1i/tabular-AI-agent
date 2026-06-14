from collections import Counter, defaultdict
from math import ceil

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.keycloak import require_authenticated_user
from app.db.session import get_db
from app.models.attempt import TransformAttempt
from app.models.metric import JobMetric
from app.schemas.metrics import MetricsSummary

router = APIRouter(prefix="/metrics", tags=["metrics"], dependencies=[Depends(require_authenticated_user)])


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, min(len(ordered) - 1, ceil(len(ordered) * 0.95) - 1))])


@router.get("/summary", response_model=MetricsSummary)
def metrics_summary(_request: object | None = None, db: Session = Depends(get_db)):
    metrics = list(db.scalars(select(JobMetric)).all())
    total = len(metrics)
    ok = sum(1 for m in metrics if m.success)
    failed = total - ok
    non_cache = [m for m in metrics if not m.cache_hit]
    first_attempt_ok = sum(1 for m in non_cache if m.success and (m.attempts or 0) == 1)
    repair_cases = [m for m in metrics if (m.attempts or 0) > 1]
    repair_ok = sum(1 for m in repair_cases if m.success)
    total_cost = sum(m.estimated_cost_usd or 0.0 for m in metrics)
    error_distribution = Counter(m.error_type for m in metrics if m.error_type)
    return MetricsSummary(
        total_jobs=total,
        successful_jobs=ok,
        failed_jobs=failed,
        success_rate=ok / total if total else 0.0,
        first_attempt_success_rate=first_attempt_ok / len(non_cache) if non_cache else 0.0,
        repair_success_rate=repair_ok / len(repair_cases) if repair_cases else 0.0,
        average_attempts=sum(m.attempts or 0 for m in metrics) / total if total else 0.0,
        average_latency_seconds=sum(m.latency_seconds or 0.0 for m in metrics) / total if total else 0.0,
        p95_latency_seconds=_p95([m.latency_seconds or 0.0 for m in metrics]),
        total_llm_calls=sum(m.llm_calls or 0 for m in metrics),
        total_tokens=sum(m.total_tokens or 0 for m in metrics),
        total_estimated_cost_usd=total_cost,
        cost_per_success=total_cost / ok if ok else 0.0,
        cache_hit_count=sum(1 for m in metrics if m.cache_hit),
        cache_hit_rate=sum(1 for m in metrics if m.cache_hit) / total if total else 0.0,
        error_type_distribution=dict(error_distribution),
        sandbox_timeout_rate=error_distribution.get("sandbox_timeout", 0) / total if total else 0.0,
        validation_failure_rate=error_distribution.get("validation_failure", 0) / total if total else 0.0,
    )


@router.get("/token-trend")
def token_trend(_request: object | None = None, db: Session = Depends(get_db)):
    attempts = list(db.scalars(select(TransformAttempt).order_by(TransformAttempt.attempt_number)).all())
    metrics = list(db.scalars(select(JobMetric)).all())
    rows_by_attempt: dict[int, dict] = defaultdict(lambda: {"attempts": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": 0.0})
    trend_source = "transform_attempts" if attempts else "job_metrics"
    if attempts:
        total_jobs = len({a.job_id for a in attempts})
        for attempt in attempts:
            row = rows_by_attempt[int(attempt.attempt_number or 0)]
            row["attempts"] += 1
            row["total_tokens"] += attempt.total_tokens or 0
            row["prompt_tokens"] += attempt.prompt_tokens or 0
            row["completion_tokens"] += attempt.completion_tokens or 0
            row["estimated_cost_usd"] += attempt.estimated_cost_usd or 0.0
    else:
        total_jobs = len(metrics)
        for metric in metrics:
            attempt_number = max(1, int(metric.attempts or 1))
            row = rows_by_attempt[attempt_number]
            row["attempts"] += 1
            row["total_tokens"] += metric.total_tokens or 0
            row["prompt_tokens"] += metric.prompt_tokens or 0
            row["completion_tokens"] += metric.completion_tokens or 0
            row["estimated_cost_usd"] += metric.estimated_cost_usd or 0.0
    cumulative = 0
    by_attempt = []
    for attempt_number in sorted(k for k in rows_by_attempt if k > 0):
        row = rows_by_attempt[attempt_number]
        cumulative += row["total_tokens"]
        count = row["attempts"] or 0
        by_attempt.append({
            "attempt_number": attempt_number,
            "attempts": count,
            "total_tokens": row["total_tokens"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "average_tokens": row["total_tokens"] / count if count else 0.0,
            "estimated_cost_usd": row["estimated_cost_usd"],
            "cumulative_tokens": cumulative,
        })
    total_attempts = sum(row["attempts"] for row in by_attempt)
    total_tokens = sum(row["total_tokens"] for row in by_attempt)
    first_avg = by_attempt[0]["average_tokens"] if by_attempt else 0.0
    last_avg = by_attempt[-1]["average_tokens"] if by_attempt else 0.0
    return {
        "total_jobs": total_jobs,
        "total_attempts": total_attempts,
        "total_tokens": total_tokens,
        "average_tokens_per_attempt": total_tokens / total_attempts if total_attempts else 0.0,
        "average_growth_ratio_first_to_last": (last_avg - first_avg) / first_avg if first_avg else 0.0,
        "trend_source": trend_source,
        "by_attempt_number": by_attempt,
        "note": "Token usage grouped by repair attempt number.",
    }
