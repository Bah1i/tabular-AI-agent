from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class AlaLensEvent(Base):
    __tablename__ = "ala_lens_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("transform_jobs.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    event_type: Mapped[str] = mapped_column(String(64), default="get")
    prompt_strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    code_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    validation_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_model_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    view_model_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameter_before_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    delta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    amendment_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameter_after_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AlaLensTypedDelta(Base):
    __tablename__ = "ala_lens_typed_deltas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("ala_lens_events.id"), nullable=True, index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("transform_jobs.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    event_type: Mapped[str] = mapped_column(String(64), default="delta", index=True)
    delta_kind: Mapped[str] = mapped_column(String(128), default="none", index=True)
    raw_error_family: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    putback_policy_name: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    putback_target: Mapped[str | None] = mapped_column(String(128), nullable=True)
    amendment_policy: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    source_mutation_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
    parameter_putback_supported: Mapped[bool] = mapped_column(Boolean, default=False)
    restoration_level: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    getput_runtime: Mapped[str | None] = mapped_column(String(64), nullable=True)
    putget_runtime: Mapped[str | None] = mapped_column(String(64), nullable=True)
    putput_runtime: Mapped[str | None] = mapped_column(String(64), nullable=True)
    semantic_signature: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    typed_delta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    putback_policy_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    lens_law_checks_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    restoration_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
