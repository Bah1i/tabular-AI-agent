from pydantic import BaseModel, Field
class MetricsSummary(BaseModel):
    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    success_rate: float
    first_attempt_success_rate: float = 0.0
    repair_success_rate: float
    average_attempts: float
    average_latency_seconds: float
    p95_latency_seconds: float = 0.0
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    cost_per_success: float = 0.0
    cache_hit_count: int = 0
    cache_hit_rate: float = 0.0
    error_type_distribution: dict[str, int] = Field(default_factory=dict)
    sandbox_timeout_rate: float = 0.0
    validation_failure_rate: float = 0.0
