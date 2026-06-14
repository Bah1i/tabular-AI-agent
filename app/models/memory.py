from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class TransformationMemory(Base):
    __tablename__ = "transformation_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source_columns_signature: Mapped[str] = mapped_column(String(2048), index=True)
    instruction: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_code: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    uses: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
