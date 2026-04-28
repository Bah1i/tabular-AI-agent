from app.db.session import SessionLocal
from app.models.job import TransformJob
from app.services.transform_service import run_transform_job
def run_transform_job_task(job_id: int) -> int:
    db = SessionLocal()
    try:
        job = db.get(TransformJob, job_id)
        if not job: raise RuntimeError(f'Job {job_id} not found')
        run_transform_job(db, job); return job_id
    finally:
        db.close()
