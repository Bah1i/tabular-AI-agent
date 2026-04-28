from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.jobs import router as jobs_router
from app.api.metrics import router as metrics_router
from app.api.ui import router as ui_router
from app.core.config import settings
from app.db.session import Base, engine
from app.models.job import TransformJob  
from app.models.metric import JobMetric  

def create_app() -> FastAPI:
    Base.metadata.create_all(bind=engine)

    app = FastAPI(title=settings.app_name)
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    app.state.templates = Jinja2Templates(directory="app/templates")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(ui_router)
    app.include_router(jobs_router)
    app.include_router(metrics_router)
    return app


app = create_app()