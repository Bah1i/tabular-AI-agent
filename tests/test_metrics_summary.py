from app.api.metrics import metrics_summary, token_trend
from app.models.attempt import TransformAttempt
from app.models.metric import JobMetric


def test_metrics_summary_includes_reliability_cost_and_error_rates(db_session):
    db_session.add_all(
        [
            JobMetric(job_id=1, success=True, attempts=1, latency_seconds=1.0, total_tokens=100, estimated_cost_usd=0.01),
            JobMetric(job_id=2, success=True, attempts=2, latency_seconds=2.0, total_tokens=200, estimated_cost_usd=0.02),
            JobMetric(
                job_id=3,
                success=False,
                attempts=3,
                latency_seconds=5.0,
                total_tokens=300,
                estimated_cost_usd=0.03,
                error_type="validation_failure",
            ),
            JobMetric(
                job_id=4,
                success=False,
                attempts=1,
                latency_seconds=10.0,
                total_tokens=50,
                estimated_cost_usd=0.005,
                error_type="sandbox_timeout",
            ),
            JobMetric(job_id=5, success=True, cache_hit=True, attempts=0, latency_seconds=0.1, total_tokens=0, estimated_cost_usd=0.0),
        ]
    )
    db_session.commit()

    summary = metrics_summary(None, db_session)

    assert summary.total_jobs == 5
    assert summary.successful_jobs == 3
    assert summary.first_attempt_success_rate == 0.25
    assert summary.repair_success_rate == 0.5
    assert summary.cache_hit_count == 1
    assert summary.cache_hit_rate == 0.2
    assert summary.cost_per_success == 0.065 / 3
    assert summary.error_type_distribution == {"validation_failure": 1, "sandbox_timeout": 1}
    assert summary.validation_failure_rate == 0.2
    assert summary.sandbox_timeout_rate == 0.2
    assert summary.p95_latency_seconds == 10.0


def test_token_trend_uses_job_totals_and_attempt_rows(db_session):
    db_session.add_all(
        [
            JobMetric(job_id=10, success=True, attempts=2, total_tokens=300, prompt_tokens=210, completion_tokens=90, estimated_cost_usd=0.03),
            JobMetric(job_id=11, success=False, attempts=1, total_tokens=120, prompt_tokens=80, completion_tokens=40, estimated_cost_usd=0.012),
            TransformAttempt(job_id=10, attempt_number=1, total_tokens=100, prompt_tokens=70, completion_tokens=30, estimated_cost_usd=0.01),
            TransformAttempt(job_id=10, attempt_number=2, total_tokens=200, prompt_tokens=140, completion_tokens=60, estimated_cost_usd=0.02),
            TransformAttempt(job_id=11, attempt_number=1, total_tokens=120, prompt_tokens=80, completion_tokens=40, estimated_cost_usd=0.012),
        ]
    )
    db_session.commit()

    trend = token_trend(None, db_session)

    assert trend["total_jobs"] == 2
    assert trend["total_attempts"] == 3
    assert trend["total_tokens"] == 420
    assert trend["trend_source"] == "transform_attempts"
    assert trend["by_attempt_number"][0]["attempt_number"] == 1
    assert trend["by_attempt_number"][0]["attempts"] == 2
    assert trend["by_attempt_number"][0]["total_tokens"] == 220
    assert trend["by_attempt_number"][1]["attempt_number"] == 2
    assert trend["by_attempt_number"][1]["average_tokens"] == 200
