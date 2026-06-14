from datetime import datetime
from sqlalchemy import Integer, Float, Boolean, ForeignKey, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base
class JobMetric(Base):
    __tablename__ = 'job_metrics'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey('transform_jobs.id'), index=True)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    latency_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    rows_processed: Mapped[int] = mapped_column(Integer, default=0)
    columns_processed: Mapped[int] = mapped_column(Integer, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
