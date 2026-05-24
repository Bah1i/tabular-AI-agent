import json
from hashlib import sha256
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.models.ala_lens import AlaLensEvent


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def source_model(df: pd.DataFrame, profile: dict) -> dict:
    return {
        "kind": "source_table",
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "schema_hash": sha256("|".join(f"{c}:{df[c].dtype}" for c in df.columns).encode("utf-8")).hexdigest()[:16],
        "profile": profile,
    }


def view_model(df: pd.DataFrame | None) -> dict | None:
    if df is None:
        return None
    return {
        "kind": "view_table",
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "sample": df.head(10).where(df.notna(), None).to_dict(orient="records"),
    }


def parameter_model(code: str | None, explanation: str | None, plan: dict | None) -> dict:
    return {
        "kind": "transformation_parameter",
        "code_hash": sha256((code or "").encode("utf-8")).hexdigest()[:16] if code else None,
        "code": code,
        "explanation": explanation,
        "plan": plan or {},
    }


def record_lens_event(
    db: Session,
    job_id: int,
    attempt_number: int,
    event_type: str,
    source: dict | None = None,
    view: dict | None = None,
    parameter_before: dict | None = None,
    delta: dict | None = None,
    amendment: dict | None = None,
    parameter_after: dict | None = None,
    note: str | None = None,
) -> None:
    db.add(
        AlaLensEvent(
            job_id=job_id,
            attempt_number=attempt_number,
            event_type=event_type,
            source_model_json=_safe_json(source) if source is not None else None,
            view_model_json=_safe_json(view) if view is not None else None,
            parameter_before_json=_safe_json(parameter_before) if parameter_before is not None else None,
            delta_json=_safe_json(delta) if delta is not None else None,
            amendment_json=_safe_json(amendment) if amendment is not None else None,
            parameter_after_json=_safe_json(parameter_after) if parameter_after is not None else None,
            note=note,
        )
    )
    db.commit()
