import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class BenchmarkStatus(str, enum.Enum):
    running = "running"
    success = "success"
    failed = "failed"


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), default="FOOFAH")
    dataset_path: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(32), default=BenchmarkStatus.running.value)
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    successful_cases: Mapped[int] = mapped_column(Integer, default=0)
    failed_cases: Mapped[int] = mapped_column(Integer, default=0)
    total_latency_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    total_estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidate_count: Mapped[int] = mapped_column(Integer, default=1)
    oracle_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    use_memory: Mapped[bool] = mapped_column(Boolean, default=True)
    benchmark_mode: Mapped[str] = mapped_column(String(64), default="strict_honest")
    memory_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    reuse_case_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    traversal_order: Mapped[str] = mapped_column(String(32), default="forward")
    benchmark_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BenchmarkCaseResult(Base):
    __tablename__ = "benchmark_case_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("benchmark_runs.id"), index=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("transform_jobs.id"), nullable=True, index=True)
    best_visible_job_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    case_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    dataset_name: Mapped[str | None] = mapped_column(String(255), default="FOOFAH", index=True)
    input_path: Mapped[str] = mapped_column(String(1024))
    output_path: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(32), default=BenchmarkStatus.running.value)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    example_success: Mapped[bool] = mapped_column(Boolean, default=False)
    generalization_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    selected_candidate_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    latency_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    token_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    prompt_strategy_used: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    hidden_judge_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
