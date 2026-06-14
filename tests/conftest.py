import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("LLM_API_KEY", "test-key")

from app.api.jobs import router as jobs_router
from app.core.config import settings
from app.db.session import Base, get_db
from app.models.ala_lens import AlaLensEvent, AlaLensTypedDelta  # noqa: F401
from app.models.attempt import TransformAttempt  # noqa: F401
from app.models.benchmark import BenchmarkCaseResult, BenchmarkRun  # noqa: F401
from app.models.job import TransformJob  # noqa: F401
from app.models.memory import TransformationMemory  # noqa: F401
from app.models.metric import JobMetric  # noqa: F401


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def api_client(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "result_dir", str(tmp_path / "results"))
    app = FastAPI()
    app.include_router(jobs_router)

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)
