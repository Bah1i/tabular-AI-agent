from contextlib import contextmanager
from app.core.config import settings
try:
    from langfuse import Langfuse
except Exception:
    Langfuse = None
def get_langfuse():
    if not settings.langfuse_enabled or not settings.langfuse_public_key or not settings.langfuse_secret_key or Langfuse is None:
        return None
    return Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key, host=settings.langfuse_host)
@contextmanager
def trace_job(job_id: int, name: str):
    lf = get_langfuse(); trace = None
    if lf:
        trace = lf.trace(name=name, metadata={'job_id': job_id}, user_id='local-user')
    try:
        yield trace
    finally:
        if lf: lf.flush()
