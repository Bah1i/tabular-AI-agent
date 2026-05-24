import enum
from datetime import datetime
from sqlalchemy import String, Text, DateTime, Enum, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base


class JobStatus(str, enum.Enum):
    created = "created"
    running = "running"
    success = "success"
    failed = "failed"


class TransformJob(Base):
    __tablename__ = "transform_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.created)
    mode: Mapped[str] = mapped_column(String(32), default="transform")

    source_filename: Mapped[str] = mapped_column(String(255))
    source_path: Mapped[str] = mapped_column(String(1024))
    expected_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    user_instruction: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_report_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    attempts: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
