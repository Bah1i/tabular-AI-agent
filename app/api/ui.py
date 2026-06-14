from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.keycloak import UserContext, require_authenticated_user
from app.core.config import settings
from app.db.session import get_db
from app.models.benchmark import BenchmarkRun
from app.models.job import JobStatus, TransformJob
from app.services.benchmark_runner import benchmark_run_overview
from app.services.job_cache import apply_cache_hit_if_available, fill_job_cache_keys
from app.services.queue import get_queue
from app.tasks import run_transform_job_task
from app.utils.files import save_upload

router = APIRouter(tags=["ui"], dependencies=[Depends(require_authenticated_user)])


@router.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db), current_user: UserContext = Depends(require_authenticated_user)):
    jobs = list(db.scalars(select(TransformJob).order_by(TransformJob.created_at.desc()).limit(20)).all())
    return request.app.state.templates.TemplateResponse(
        "index.html",
        {"request": request, "jobs": jobs, "current_user": current_user, "keycloak_enabled": settings.keycloak_enabled},
    )


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
    fill_job_cache_keys(job)
    db.add(job)
    db.commit()
    db.refresh(job)
    if not apply_cache_hit_if_available(db, job):
        get_queue().enqueue(run_transform_job_task, job.id, job_timeout=1200)
    return RedirectResponse(url=f"/ui/jobs/{job.id}", status_code=303)


@router.get("/ui/jobs/{job_id}", response_class=HTMLResponse)
def show_job(request: Request, job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return request.app.state.templates.TemplateResponse("job.html", {"request": request, "job": job})


@router.get("/ui/metrics", response_class=HTMLResponse)
def metrics_page(request: Request):
    return request.app.state.templates.TemplateResponse("metrics.html", {"request": request})


@router.get("/ui/postgres", response_class=HTMLResponse)
def postgres_page(request: Request):
    return request.app.state.templates.TemplateResponse("postgres.html", {"request": request})


@router.get("/ui/benchmarks", response_class=HTMLResponse)
def benchmarks_page(request: Request, page: int = 1, db: Session = Depends(get_db)):
    per_page = 30
    page = max(1, page)
    total_runs = int(db.scalar(select(func.count()).select_from(BenchmarkRun)) or 0)
    total_pages = max(1, (total_runs + per_page - 1) // per_page)
    page = min(page, total_pages)
    offset = (page - 1) * per_page
    runs = list(
        db.scalars(
            select(BenchmarkRun)
            .order_by(BenchmarkRun.created_at.desc(), BenchmarkRun.id.desc())
            .offset(offset)
            .limit(per_page)
        ).all()
    )
    run_rows = []
    for run in runs:
        try:
            run_rows.append(benchmark_run_overview(db, run.id))
        except Exception:
            run_rows.append({
                "run_id": run.id,
                "name": run.name,
                "benchmark_label": run.benchmark_label,
                "benchmark_mode": run.benchmark_mode,
                "mode_label": run.benchmark_mode,
                "candidate_count": run.candidate_count,
                "memory_enabled": run.memory_enabled if run.memory_enabled is not None else run.name.upper() == "FOOFAH",
                "reuse_case_enabled": bool(run.reuse_case_enabled),
                "traversal_order": run.traversal_order,
                "cache_hit_cases": 0,
                "cache_hit_rate": 0.0,
                "status": run.status,
                "total_cases": run.total_cases,
                "job_successful_cases": 0,
                "benchmark_successful_cases": run.successful_cases,
                "successful_cases": run.successful_cases,
                "failed_cases": run.failed_cases,
                "generalization_success_rate": None,
                "total_latency_seconds": run.total_latency_seconds,
                "total_estimated_cost_usd": run.total_estimated_cost_usd,
            })
    return request.app.state.templates.TemplateResponse(
        "benchmarks.html",
        {
            "request": request,
            "run_rows": run_rows,
            "runs_page": page,
            "runs_total_pages": total_pages,
            "runs_total": total_runs,
            "runs_per_page": per_page,
            "runs_has_prev": page > 1,
            "runs_has_next": page < total_pages,
        },
    )
