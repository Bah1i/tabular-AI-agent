from pydantic import BaseModel
class MetricsSummary(BaseModel):
    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    success_rate: float
    repair_success_rate: float
    average_attempts: float
    average_latency_seconds: float
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
