from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.job import JobStatus, TransformJob
from app.services.queue import get_queue
from app.tasks import run_transform_job_task
from app.utils.files import save_upload

router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    jobs = list(db.scalars(select(TransformJob).order_by(TransformJob.created_at.desc()).limit(20)).all())
    return request.app.state.templates.TemplateResponse("index.html", {"request": request, "jobs": jobs})


@router.post("/ui/jobs")
async def create_job_from_ui(
    request: Request,
    source_file: UploadFile = File(...),
    expected_file: UploadFile | None = File(default=None),
    instruction: str | None = Form(default=None),
    mode: str = Form(default="transform"),
    db: Session = Depends(get_db),
):
    if mode not in {"transform", "query"}:
        raise HTTPException(status_code=400, detail="Unknown job mode")
    expected_path = await save_upload(expected_file, settings.upload_dir) if expected_file and expected_file.filename else None
    if mode == "transform" and not expected_path:
        raise HTTPException(status_code=400, detail="Expected example is required for transform mode")
    job = TransformJob(
        source_filename=source_file.filename or "source",
        source_path=await save_upload(source_file, settings.upload_dir),
        expected_path=expected_path,
        user_instruction=instruction,
        status=JobStatus.running,
        mode=mode,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    get_queue().enqueue(run_transform_job_task, job.id, job_timeout=1200)
    return RedirectResponse(url=f"/ui/jobs/{job.id}", status_code=303)


@router.get("/ui/jobs/{job_id}", response_class=HTMLResponse)
def show_job(request: Request, job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return request.app.state.templates.TemplateResponse("job.html", {"request": request, "job": job})
