from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.db.session import get_db
from app.models.metric import JobMetric
from app.schemas.metrics import MetricsSummary
router = APIRouter(prefix='/metrics', tags=['metrics'])
@router.get('/summary', response_model=MetricsSummary)
def metrics_summary(db: Session = Depends(get_db)):
    metrics = list(db.scalars(select(JobMetric)).all()); total = len(metrics); ok = sum(1 for m in metrics if m.success); failed = total - ok
    repair_cases = sum(1 for m in metrics if m.attempts > 1); repair_ok = sum(1 for m in metrics if m.success and m.attempts > 1)
    return MetricsSummary(total_jobs=total, successful_jobs=ok, failed_jobs=failed, success_rate=ok/total if total else 0.0, repair_success_rate=repair_ok/repair_cases if repair_cases else 0.0, average_attempts=sum(m.attempts for m in metrics)/total if total else 0.0, average_latency_seconds=sum(m.latency_seconds for m in metrics)/total if total else 0.0)
