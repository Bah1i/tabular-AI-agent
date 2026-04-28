from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import get_db
from app.models.job import TransformJob, JobStatus
from app.schemas.job import JobCreateResponse, JobRead, RunJobResponse
from app.services.queue import get_queue
from app.services.transform_service import run_transform_job
from app.tasks import run_transform_job_task
from app.utils.files import save_upload
router = APIRouter(prefix='/jobs', tags=['jobs'])
@router.post('', response_model=JobCreateResponse)
async def create_job(source_file: UploadFile = File(...), expected_file: UploadFile = File(...), instruction: str | None = Form(default=None), db: Session = Depends(get_db)):
    job = TransformJob(source_filename=source_file.filename or 'source', source_path=await save_upload(source_file, settings.upload_dir), expected_path=await save_upload(expected_file, settings.upload_dir), user_instruction=instruction)
    db.add(job); db.commit(); db.refresh(job); return JobCreateResponse(job_id=job.id, status=job.status.value)
@router.post('/{job_id}/enqueue', response_model=RunJobResponse)
def enqueue_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    get_queue().enqueue(run_transform_job_task, job.id, job_timeout=600); job.status = JobStatus.running; db.commit(); return RunJobResponse(job_id=job.id, status=job.status.value, attempts=job.attempts, result_path=job.result_path, error_message=job.error_message)
@router.post('/{job_id}/run', response_model=RunJobResponse)
def run_job_sync(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    job = run_transform_job(db, job); return RunJobResponse(job_id=job.id, status=job.status.value, attempts=job.attempts, result_path=job.result_path, error_message=job.error_message)
@router.get('/{job_id}', response_model=JobRead)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    return job
@router.get('/{job_id}/result')
def download_result(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job or not job.result_path: raise HTTPException(status_code=404, detail='Result not found')
    return FileResponse(path=job.result_path, media_type='text/csv', filename=f'job_{job.id}_result.csv')
