from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.auth import router as auth_router
from app.api.benchmarks import router as benchmarks_router
from app.api.jobs import router as jobs_router
from app.api.metrics import router as metrics_router
from app.api.postgres import router as postgres_router
from app.api.ui import router as ui_router
from app.core.config import settings
from app.db.migrations import run_lightweight_migrations
from app.db.session import Base, engine
from app.models.ala_lens import AlaLensEvent, AlaLensTypedDelta  # noqa: F401
from app.models.attempt import TransformAttempt  # noqa: F401
from app.models.benchmark import BenchmarkCaseResult, BenchmarkRun  # noqa: F401
from app.models.job import TransformJob  # noqa: F401
from app.models.memory import TransformationMemory  # noqa: F401
from app.models.metric import JobMetric  # noqa: F401


def create_app() -> FastAPI:
    Base.metadata.create_all(bind=engine)
    run_lightweight_migrations()

    app = FastAPI(title=settings.app_name)
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    app.state.templates = Jinja2Templates(directory="app/templates")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(auth_router)
    app.include_router(ui_router)
    app.include_router(jobs_router)
    app.include_router(benchmarks_router)
    app.include_router(metrics_router)
    app.include_router(postgres_router)
    return app


app = create_app()
