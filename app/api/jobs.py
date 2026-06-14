from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import get_db
from app.models.ala_lens import AlaLensEvent, AlaLensTypedDelta
from app.models.benchmark import BenchmarkCaseResult
from app.models.job import TransformJob, JobStatus
from app.models.attempt import TransformAttempt
from app.schemas.job import JobCreateResponse, JobRead, RunJobResponse
from app.services.queue import get_queue
from app.services.profiler import dataframe_analysis, json_safe, read_table
from app.services.table_view import table_preview
from app.services.job_cache import apply_cache_hit_if_available, fill_job_cache_keys
from app.services.ala_lens import lens_delta_statistics
from app.services.transform_service import run_transform_job
from app.tasks import run_transform_job_task
from app.utils.files import save_upload
router = APIRouter(prefix='/jobs', tags=['jobs'])
@router.post('', response_model=JobCreateResponse)
async def create_job(source_file: UploadFile = File(...), expected_file: UploadFile | None = File(default=None), instruction: str | None = Form(default=None), mode: str = Form(default='transform'), db: Session = Depends(get_db)):
    expected_path = await save_upload(expected_file, settings.upload_dir) if expected_file and expected_file.filename else None
    if mode == 'transform' and not expected_path: raise HTTPException(status_code=400, detail='Expected file is required for transform mode')
    job = TransformJob(source_filename=source_file.filename or 'source', source_path=await save_upload(source_file, settings.upload_dir), expected_path=expected_path, user_instruction=instruction, mode=mode)
    fill_job_cache_keys(job)
    db.add(job); db.commit(); db.refresh(job); return JobCreateResponse(job_id=job.id, status=job.status.value)
@router.post('/{job_id}/enqueue', response_model=RunJobResponse)
def enqueue_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    if not job.source_hash:
        fill_job_cache_keys(job); db.commit(); db.refresh(job)
    if apply_cache_hit_if_available(db, job):
        return RunJobResponse(job_id=job.id, status=job.status.value, attempts=job.attempts, result_path=job.result_path, error_message=job.error_message)
    get_queue().enqueue(run_transform_job_task, job.id, job_timeout=600); job.status = JobStatus.running; db.commit(); return RunJobResponse(job_id=job.id, status=job.status.value, attempts=job.attempts, result_path=job.result_path, error_message=job.error_message)
@router.post('/{job_id}/run', response_model=RunJobResponse)
def run_job_sync(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    if not job.source_hash:
        fill_job_cache_keys(job); db.commit(); db.refresh(job)
    if apply_cache_hit_if_available(db, job):
        return RunJobResponse(job_id=job.id, status=job.status.value, attempts=job.attempts, result_path=job.result_path, error_message=job.error_message)
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
    benchmark_case = db.scalars(select(BenchmarkCaseResult).where(BenchmarkCaseResult.job_id == job.id).order_by(BenchmarkCaseResult.created_at.desc(), BenchmarkCaseResult.id.desc()).limit(1)).first()
    benchmark_case_payload = None
    if benchmark_case:
        benchmark_case_payload = {'id': benchmark_case.id, 'run_id': benchmark_case.run_id, 'case_name': benchmark_case.case_name, 'status': benchmark_case.status, 'success': benchmark_case.success, 'error_message': benchmark_case.error_message}
    return {'id': job.id, 'mode': job.mode, 'prompt_strategy': job.prompt_strategy, 'status': job.status.value, 'attempts': job.attempts, 'error_message': job.error_message, 'explanation': job.explanation, 'generated_code': job.generated_code, 'result_path': job.result_path, 'validation_report_json': job.validation_report_json, 'cache_hit_from_job_id': job.cache_hit_from_job_id, 'benchmark_case': benchmark_case_payload, 'attempt_history': [{'attempt': a.attempt_number, 'prompt_strategy': a.prompt_strategy, 'success': a.success, 'error_message': a.error_message, 'explanation': a.explanation, 'total_tokens': a.total_tokens, 'estimated_cost_usd': a.estimated_cost_usd} for a in attempts]}
@router.get('/{job_id}/preview/{kind}')
def preview_job_table(job_id: int, kind: str, limit: int = 100, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    paths = {'source': job.source_path, 'expected': job.expected_path, 'result': job.result_path}
    path = paths.get(kind)
    if not path: raise HTTPException(status_code=404, detail='Table not found')
    return table_preview(path, limit, headerless=job.prompt_strategy == 'foofah' and kind in {'source', 'expected'})
@router.get('/{job_id}/analysis')
def analyze_job_source(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    return json_safe(dataframe_analysis(read_table(job.source_path, headerless=job.prompt_strategy == 'foofah')))
@router.get('/{job_id}/ala-lens')
def get_ala_lens_events(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    events = list(db.scalars(select(AlaLensEvent).where(AlaLensEvent.job_id == job.id).order_by(AlaLensEvent.id)).all())
    typed_deltas = list(db.scalars(select(AlaLensTypedDelta).where(AlaLensTypedDelta.job_id == job.id).order_by(AlaLensTypedDelta.id)).all())
    return {
        'job_id': job.id,
        'lens_delta_statistics': lens_delta_statistics(db, [job.id]),
        'events': [{'id': e.id, 'attempt': e.attempt_number, 'type': e.event_type, 'prompt_strategy': e.prompt_strategy, 'code_hash': e.code_hash, 'validation_status': e.validation_status, 'source_model': e.source_model_json, 'view_model': e.view_model_json, 'parameter_before': e.parameter_before_json, 'delta': e.delta_json, 'amendment': e.amendment_json, 'parameter_after': e.parameter_after_json, 'note': e.note, 'created_at': e.created_at.isoformat()} for e in events],
        'typed_deltas': [
            {
                'id': d.id,
                'event_id': d.event_id,
                'attempt': d.attempt_number,
                'event_type': d.event_type,
                'delta_kind': d.delta_kind,
                'raw_error_family': d.raw_error_family,
                'confidence': d.confidence,
                'putback_policy_name': d.putback_policy_name,
                'putback_target': d.putback_target,
                'amendment_policy': d.amendment_policy,
                'source_mutation_allowed': d.source_mutation_allowed,
                'parameter_putback_supported': d.parameter_putback_supported,
                'restoration_level': d.restoration_level,
                'getput_runtime': d.getput_runtime,
                'putget_runtime': d.putget_runtime,
                'putput_runtime': d.putput_runtime,
                'semantic_signature': d.semantic_signature,
                'typed_delta_json': d.typed_delta_json,
                'putback_policy_json': d.putback_policy_json,
                'lens_law_checks_json': d.lens_law_checks_json,
                'restoration_json': d.restoration_json,
                'created_at': d.created_at.isoformat(),
            }
            for d in typed_deltas
        ],
    }
@router.get('/{job_id}/result')
def download_result(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TransformJob, job_id)
    if not job or not job.result_path: raise HTTPException(status_code=404, detail='Result not found')
    return FileResponse(path=job.result_path, media_type='text/csv', filename=f'job_{job.id}_result.csv')
