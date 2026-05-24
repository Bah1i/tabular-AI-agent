from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class AlaLensEvent(Base):
    __tablename__ = "ala_lens_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("transform_jobs.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    event_type: Mapped[str] = mapped_column(String(64), default="get")
    source_model_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    view_model_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameter_before_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    delta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    amendment_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameter_after_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
