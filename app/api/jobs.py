from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import get_db
from app.models.ala_lens import AlaLensEvent
from app.models.job import TransformJob, JobStatus
from app.models.attempt import TransformAttempt
from app.schemas.job import JobCreateResponse, JobRead, RunJobResponse
from app.services.queue import get_queue
from app.services.profiler import dataframe_analysis, json_safe, read_table
from app.services.table_view import table_preview
from app.services.transform_service import run_transform_job
from app.tasks import run_transform_job_task
from app.utils.files import save_upload
router = APIRouter(prefix='/jobs', tags=['jobs'])
@router.post('', response_model=JobCreateResponse)
async def create_job(source_file: UploadFile = File(...), expected_file: UploadFile | None = File(default=None), instruction: str | None = Form(default=None), mode: str = Form(default='transform'), db: Session = Depends(get_db)):
    expected_path = await save_upload(expected_file, settings.upload_dir) if expected_file and expected_file.filename else None
    if mode == 'transform' and not expected_path: raise HTTPException(status_code=400, detail='Expected file is required for transform mode')
    job = TransformJob(source_filename=source_file.filename or 'source', source_path=await save_upload(source_file, settings.upload_dir), expected_path=expected_path, user_instruction=instruction, mode=mode)
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
@router.get('/{job_id}/status')
def get_job_status(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    attempts = list(db.scalars(select(TransformAttempt).where(TransformAttempt.job_id == job.id).order_by(TransformAttempt.attempt_number)).all())
    return {'id': job.id, 'mode': job.mode, 'status': job.status.value, 'attempts': job.attempts, 'error_message': job.error_message, 'explanation': job.explanation, 'generated_code': job.generated_code, 'result_path': job.result_path, 'validation_report_json': job.validation_report_json, 'attempt_history': [{'attempt': a.attempt_number, 'success': a.success, 'error_message': a.error_message, 'explanation': a.explanation, 'total_tokens': a.total_tokens, 'estimated_cost_usd': a.estimated_cost_usd} for a in attempts]}
@router.get('/{job_id}/preview/{kind}')
def preview_job_table(job_id: int, kind: str, limit: int = 100, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    paths = {'source': job.source_path, 'expected': job.expected_path, 'result': job.result_path}
    path = paths.get(kind)
    if not path: raise HTTPException(status_code=404, detail='Table not found')
    return table_preview(path, limit)
@router.get('/{job_id}/analysis')
def analyze_job_source(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    return json_safe(dataframe_analysis(read_table(job.source_path)))
@router.get('/{job_id}/ala-lens')
def get_ala_lens_events(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    events = list(db.scalars(select(AlaLensEvent).where(AlaLensEvent.job_id == job.id).order_by(AlaLensEvent.id)).all())
    return {'job_id': job.id, 'events': [{'id': e.id, 'attempt': e.attempt_number, 'type': e.event_type, 'source_model': e.source_model_json, 'view_model': e.view_model_json, 'parameter_before': e.parameter_before_json, 'delta': e.delta_json, 'amendment': e.amendment_json, 'parameter_after': e.parameter_after_json, 'note': e.note, 'created_at': e.created_at.isoformat()} for e in events]}
@router.get('/{job_id}/result')
def download_result(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job or not job.result_path: raise HTTPException(status_code=404, detail='Result not found')
    return FileResponse(path=job.result_path, media_type='text/csv', filename=f'job_{job.id}_result.csv')
