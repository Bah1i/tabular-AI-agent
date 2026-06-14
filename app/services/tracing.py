from contextlib import contextmanager
from typing import Any

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
def trace_job(job_id: int, name: str, metadata: dict[str, Any] | None = None):
    lf = get_langfuse()
    trace = None
    if lf:
        trace_metadata = {"job_id": job_id, **(metadata or {})}
        trace = lf.trace(name=name, metadata=trace_metadata, user_id="local-user", session_id=f"job-{job_id}")
    try:
        yield trace
    finally:
        if lf:
            lf.flush()


@contextmanager
def trace_span(trace, name: str, metadata: dict[str, Any] | None = None, input_data: Any = None):
    span = None
    if trace:
        try:
            span = trace.span(name=name, metadata=metadata or {}, input=input_data)
        except Exception:
            span = None
    try:
        yield span
        if span:
            try:
                span.end(output={"status": "ok"})
            except Exception:
                pass
    except Exception as exc:
        if span:
            try:
                span.end(output={"status": "error", "error": str(exc)})
            except Exception:
                pass
        raise


def score_trace(trace, name: str, value: float, comment: str | None = None) -> None:
    if not trace:
        return
    try:
        trace.score(name=name, value=value, comment=comment)
    except Exception:
        pass


def update_trace(trace, metadata: dict[str, Any] | None = None, output: Any = None) -> None:
    if not trace:
        return
    try:
        trace.update(metadata=metadata or {}, output=output)
    except Exception:
        pass
