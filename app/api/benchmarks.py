from __future__ import annotations

import threading

from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session

from app.auth.keycloak import require_admin_user
from app.db.session import SessionLocal, get_db
from app.services.benchmark_runner import BenchmarkCase, BenchmarkRunner, benchmark_run_overview, benchmark_summary

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"], dependencies=[Depends(require_admin_user)])


def _run_foofah_background(
    run_id: int,
    cases: list[BenchmarkCase],
    candidate_count: int,
    oracle_mode: bool,
    memory_enabled: bool,
    reuse_case_enabled: bool,
) -> None:
    db = SessionLocal()
    try:
        BenchmarkRunner(db).process_run(
            run_id,
            cases,
            foofah_candidate_count=candidate_count,
            oracle_mode=oracle_mode,
            memory_enabled=memory_enabled,
            reuse_case_enabled=reuse_case_enabled,
        )
    finally:
        db.close()


@router.post("/foofah/start")
def start_foofah_benchmark(
    max_cases: int | None = Form(default=None),
    case_block: int | None = Form(default=None),
    candidate_count: int = Form(default=1),
    memory_enabled: bool = Form(default=True),
    reuse_case_enabled: bool = Form(default=False),
    oracle_mode: bool = Form(default=False),
    db: Session = Depends(get_db),
):
    try:
        run, cases = BenchmarkRunner(db).create_foofah_run(
            max_cases=max_cases,
            case_block=case_block or None,
            candidate_count=candidate_count,
            oracle_mode=oracle_mode,
            memory_enabled=memory_enabled,
            reuse_case_enabled=reuse_case_enabled,
        )
        thread = threading.Thread(
            target=_run_foofah_background,
            args=(run.id, cases, candidate_count, oracle_mode, memory_enabled, reuse_case_enabled),
            daemon=True,
        )
        thread.start()
        return benchmark_run_overview(db, run.id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/foofah")
def run_foofah_benchmark_sync(
    max_cases: int | None = Form(default=None),
    case_block: int | None = Form(default=None),
    candidate_count: int = Form(default=1),
    memory_enabled: bool = Form(default=True),
    reuse_case_enabled: bool = Form(default=False),
    oracle_mode: bool = Form(default=False),
    db: Session = Depends(get_db),
):
    try:
        run = BenchmarkRunner(db).run_foofah(
            max_cases=max_cases,
            case_block=case_block or None,
            candidate_count=candidate_count,
            oracle_mode=oracle_mode,
            memory_enabled=memory_enabled,
            reuse_case_enabled=reuse_case_enabled,
        )
        return benchmark_summary(db, run.id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{run_id}/summary")
def get_benchmark_summary(run_id: int, db: Session = Depends(get_db)):
    try:
        return benchmark_summary(db, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/overview")
def get_benchmark_overview(run_id: int, db: Session = Depends(get_db)):
    try:
        return benchmark_run_overview(db, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
