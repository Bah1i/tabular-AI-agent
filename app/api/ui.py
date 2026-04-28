from fastapi import APIRouter, Depends, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import get_db
from app.models.job import TransformJob, JobStatus
from app.services.queue import get_queue
from app.tasks import run_transform_job_task
from app.utils.files import save_upload
router = APIRouter(tags=['ui'])
@router.get('/', response_class=HTMLResponse)
def index(request: Request): return request.app.state.templates.TemplateResponse('index.html', {'request': request})
@router.post('/ui/jobs')
async def create_job_from_ui(request: Request, source_file: UploadFile = File(...), expected_file: UploadFile = File(...), instruction: str | None = Form(default=None), db: Session = Depends(get_db)):
    job = TransformJob(source_filename=source_file.filename or 'source', source_path=await save_upload(source_file, settings.upload_dir), expected_path=await save_upload(expected_file, settings.upload_dir), user_instruction=instruction, status=JobStatus.running)
    db.add(job); db.commit(); db.refresh(job); get_queue().enqueue(run_transform_job_task, job.id, job_timeout=600); return RedirectResponse(url=f'/ui/jobs/{job.id}', status_code=303)
@router.get('/ui/jobs/{job_id}', response_class=HTMLResponse)
def show_job(request: Request, job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    return request.app.state.templates.TemplateResponse('job.html', {'request': request, 'job': job})
