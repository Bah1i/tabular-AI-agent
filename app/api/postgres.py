import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.auth.keycloak import UserContext, require_admin_user
from app.core.config import settings
from app.services.comparator import compare_dataframes_report
from app.services.llm_client import get_llm_client
from app.services.postgres_readonly import (
    explain_read_only_query,
    get_schema_metadata,
    list_databases,
    list_schemas,
    list_tables,
    preview_table_rows,
    register_connection,
    run_read_only_query,
)
from app.services.prompts import SQL_SYSTEM_PROMPT, build_postgres_sql_from_expected_prompt, build_postgres_sql_prompt
from app.services.profiler import read_table
from app.services.sql_validator import SQLValidationError, ensure_limit
from app.utils.files import save_upload

router = APIRouter(prefix="/postgres", tags=["postgres"], dependencies=[Depends(require_admin_user)])

SQL_EXPECTED_MAX_ATTEMPTS = 3
SQL_EXPECTED_SAMPLE_ROWS = 25


def _clean_form_text(value: str | None) -> str:
    return (value or "").strip()


def _expected_records(expected_df: pd.DataFrame) -> list[dict]:
    head = expected_df.head(SQL_EXPECTED_SAMPLE_ROWS).astype(object)
    sample = head.where(pd.notna(head), None)
    return sample.to_dict(orient="records")


def _empty_preview(sql: str = "") -> dict:
    return {"sql": sql, "columns": [], "rows": [], "row_count": 0}


def _dataframe_preview(df: pd.DataFrame, limit: int = 50, offset: int = 0) -> dict:
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    sample = df.iloc[offset: offset + limit].astype(object)
    sample = sample.where(pd.notna(sample), None)
    return {
        "columns": [str(col) for col in df.columns],
        "rows": sample.to_dict(orient="records"),
        "row_count": int(len(sample)),
        "total_rows": int(len(df)),
        "limit": limit,
        "offset": offset,
    }


def _validate_preview_against_expected(preview: dict, expected_df: pd.DataFrame) -> dict:
    actual_df = pd.DataFrame(preview["rows"], columns=preview["columns"])
    actual_sample = actual_df.head(len(expected_df))
    report = compare_dataframes_report(actual_sample, expected_df)
    return {
        "ok": report.ok,
        "message": report.message,
        "details": report.details,
        "expected_rows": int(len(expected_df)),
        "compared_rows": int(len(actual_sample)),
    }


def _sql_result_status(execution_ok: bool, expected_validation: dict | None) -> str:
    if not execution_ok:
        return "execution_failed"
    if expected_validation is None:
        return "executed"
    return "expected_match" if expected_validation.get("ok") else "expected_mismatch"


def _sql_lens_payload(
    *,
    source: str,
    sql: str,
    preview: dict,
    expected_validation: dict | None,
    attempts: list[dict],
    execution_ok: bool,
) -> dict:
    expected_match = expected_validation.get("ok") if expected_validation is not None else None
    return {
        "status": _sql_result_status(execution_ok, expected_validation),
        "forward_transform": {
            "source": source,
            "parameter": "read_only_sql",
            "sql_preview": sql,
            "rows": int(preview.get("row_count") or len(preview.get("rows") or [])),
            "columns": list(preview.get("columns") or []),
        },
        "delta_discovery": {
            "kind": "none" if expected_validation is None or expected_match else "expected_value_mismatch",
            "expected_match": expected_match,
            "message": expected_validation.get("message") if expected_validation else "No expected file was attached.",
        },
        "repair_or_retry": {
            "attempts": len(attempts),
            "policy": "retry_sql_generation_against_expected" if expected_validation is not None and not expected_match else "no_retry_needed",
        },
        "restoration": {
            "read_only_execution": "ok" if execution_ok else "failed",
            "expected_restored": expected_match,
            "note": (
                "SQL executed successfully; expected file differs from live database result."
                if execution_ok and expected_validation is not None and not expected_match
                else "SQL executed successfully."
                if execution_ok
                else "SQL did not execute."
            ),
        },
    }


def _sql_response(
    *,
    sql: str,
    explanation: str,
    preview: dict,
    plan: dict,
    metadata: dict,
    expected_df: pd.DataFrame | None,
    expected_validation: dict | None,
    sql_attempts: list[dict],
    source: str,
    execution_ok: bool = True,
) -> dict:
    return {
        "sql": sql,
        "explanation": explanation,
        "preview": preview,
        "explain": plan,
        "metadata": metadata,
        "expected_preview": _dataframe_preview(expected_df) if expected_df is not None else None,
        "expected_validation": expected_validation,
        "sql_attempts": sql_attempts,
        "attempt_count": len(sql_attempts),
        "execution_ok": execution_ok,
        "expected_match": expected_validation.get("ok") if expected_validation is not None else None,
        "result_status": _sql_result_status(execution_ok, expected_validation),
        "sql_lens": _sql_lens_payload(
            source=source,
            sql=sql,
            preview=preview,
            expected_validation=expected_validation,
            attempts=sql_attempts,
            execution_ok=execution_ok,
        ),
    }


@router.post("/connect")
def connect_postgres(
    host: str = Form(...),
    port: int = Form(5432),
    username: str = Form(...),
    password: str = Form(...),
    _: UserContext = Depends(require_admin_user),
):
    try:
        connection_id = register_connection(host, port, username, password)
        databases = list_databases(connection_id)
        return {"connection_id": connection_id, "databases": databases}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not connect to PostgreSQL: {type(exc).__name__}") from exc


@router.get("/{connection_id}/schemas")
def get_schemas(connection_id: str, database: str):
    try:
        return {"schemas": list_schemas(connection_id, database)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not list schemas: {type(exc).__name__}") from exc


@router.get("/{connection_id}/tables")
def get_tables(connection_id: str, database: str, schema: str):
    try:
        return {"tables": list_tables(connection_id, database, schema)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not list tables: {type(exc).__name__}") from exc


@router.get("/{connection_id}/metadata")
def get_metadata(connection_id: str, database: str, schema: str, table: str | None = None):
    try:
        return get_schema_metadata(connection_id, database, schema, table)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load metadata: {type(exc).__name__}") from exc


@router.get("/{connection_id}/table-preview")
def get_table_preview(
    connection_id: str,
    database: str,
    schema: str,
    table: str,
    limit: int = 50,
    offset: int = 0,
):
    try:
        if not table:
            raise HTTPException(status_code=400, detail="Select one table before loading preview.")
        return preview_table_rows(connection_id, database, schema, table, limit=limit, offset=offset)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not preview table: {type(exc).__name__}") from exc


@router.post("/expected-preview")
async def preview_expected_file(
    expected_file: UploadFile = File(...),
    limit: int = Form(default=50),
    offset: int = Form(default=0),
):
    try:
        expected_path = await save_upload(expected_file, settings.upload_dir)
        expected_df = read_table(expected_path)
        preview = _dataframe_preview(expected_df, limit=limit, offset=offset)
        preview["filename"] = expected_file.filename
        return preview
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not preview expected file: {type(exc).__name__}") from exc


@router.post("/{connection_id}/query")
async def run_query(
    connection_id: str,
    database: str = Form(...),
    schema: str = Form(...),
    table: str | None = Form(default=None),
    natural_language_query: str | None = Form(default=None),
    sql_query: str | None = Form(default=None),
    expected_file: UploadFile | None = File(default=None),
    statement_timeout_ms: int = Form(default=5000),
):
    try:
        table = table or None
        natural_language_query = _clean_form_text(natural_language_query)
        sql_query = _clean_form_text(sql_query)
        metadata = get_schema_metadata(connection_id, database, schema, table)
        expected_df = None
        if expected_file and expected_file.filename:
            expected_path = await save_upload(expected_file, settings.upload_dir)
            expected_df = read_table(expected_path)

        if sql_query:
            sql = sql_query
            explanation = "Manual SQL query."
            limited_sql = ensure_limit(sql)
            preview = run_read_only_query(connection_id, database, limited_sql, statement_timeout_ms)
            plan = explain_read_only_query(connection_id, database, limited_sql, statement_timeout_ms)
            expected_validation = _validate_preview_against_expected(preview, expected_df) if expected_df is not None else None
            sql_attempts = [
                {
                    "attempt": 1,
                    "source": "manual_sql",
                    "success": expected_validation["ok"] if expected_validation else True,
                    "execution_success": True,
                    "validation_success": expected_validation["ok"] if expected_validation else None,
                    "status": _sql_result_status(True, expected_validation),
                    "sql": limited_sql,
                    "explanation": explanation,
                    "error": None,
                    "validation": expected_validation,
                }
            ]
            return _sql_response(
                sql=limited_sql,
                explanation=explanation,
                preview=preview,
                plan=plan,
                metadata=metadata,
                expected_df=expected_df,
                expected_validation=expected_validation,
                sql_attempts=sql_attempts,
                source="manual_sql",
                execution_ok=True,
            )

        elif natural_language_query:
            if expected_df is None:
                prompt = build_postgres_sql_prompt(metadata, natural_language_query)
                result = get_llm_client().generate_sql(
                    prompt,
                    generation_name="postgres-nl-to-sql",
                    system_prompt=SQL_SYSTEM_PROMPT,
                )
                sql = result.sql
                explanation = result.explanation
                limited_sql = ensure_limit(sql)
                preview = run_read_only_query(connection_id, database, limited_sql, statement_timeout_ms)
                plan = explain_read_only_query(connection_id, database, limited_sql, statement_timeout_ms)
                sql_attempts = [
                    {
                        "attempt": 1,
                        "source": "natural_language",
                        "success": True,
                        "execution_success": True,
                        "validation_success": None,
                        "status": "executed",
                        "sql": limited_sql,
                        "explanation": explanation,
                        "error": None,
                        "tokens": {
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "total_tokens": result.total_tokens,
                            "estimated_cost_usd": result.estimated_cost_usd,
                        },
                    }
                ]
                return _sql_response(
                    sql=limited_sql,
                    explanation=explanation,
                    preview=preview,
                    plan=plan,
                    metadata=metadata,
                    expected_df=None,
                    expected_validation=None,
                    sql_attempts=sql_attempts,
                    source="natural_language",
                    execution_ok=True,
                )
        else:
            if expected_df is None:
                raise HTTPException(status_code=400, detail="Provide natural_language_query, sql_query, or an expected_file for expected-only SQL synthesis.")

        previous_attempts: list[dict] = []
        sql_attempts: list[dict] = []
        preview = _empty_preview()
        plan = {"sql": "", "plan": []}
        expected_validation = {
            "ok": False,
            "message": "SQL was not generated.",
            "details": {},
            "expected_rows": int(len(expected_df)) if expected_df is not None else 0,
            "compared_rows": 0,
        }
        limited_sql = ""
        explanation = ""
        expected_rows = _expected_records(expected_df)
        source = "natural_language_expected" if natural_language_query else "expected_only"
        client = get_llm_client()
        for attempt in range(1, SQL_EXPECTED_MAX_ATTEMPTS + 1):
            prompt = build_postgres_sql_from_expected_prompt(
                metadata,
                expected_rows,
                natural_language_query=natural_language_query,
                previous_attempts=previous_attempts,
            )
            result = client.generate_sql(
                prompt,
                generation_name=f"postgres-expected-to-sql-attempt-{attempt}",
                system_prompt=SQL_SYSTEM_PROMPT,
            )
            sql = result.sql
            explanation = result.explanation
            limited_sql = sql
            preview = _empty_preview(sql)
            plan = {"sql": "", "plan": []}
            attempt_payload = {
                "attempt": attempt,
                "source": source,
                "success": False,
                "sql": sql,
                "explanation": explanation,
                "error": None,
                "validation": None,
                "tokens": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "estimated_cost_usd": result.estimated_cost_usd,
                },
            }
            try:
                limited_sql = ensure_limit(sql)
                preview = run_read_only_query(connection_id, database, limited_sql, statement_timeout_ms)
                plan = explain_read_only_query(connection_id, database, limited_sql, statement_timeout_ms)
                expected_validation = _validate_preview_against_expected(preview, expected_df)
                attempt_payload["sql"] = limited_sql
                attempt_payload["validation"] = expected_validation
                attempt_payload["execution_success"] = True
                attempt_payload["validation_success"] = bool(expected_validation["ok"])
                attempt_payload["status"] = _sql_result_status(True, expected_validation)
                attempt_payload["success"] = bool(expected_validation["ok"])
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                attempt_payload["error"] = message
                attempt_payload["execution_success"] = False
                attempt_payload["validation_success"] = None
                attempt_payload["status"] = "execution_failed"
                expected_validation = {
                    "ok": False,
                    "message": message,
                    "details": {},
                    "expected_rows": int(len(expected_df)),
                    "compared_rows": 0,
                }

            sql_attempts.append(attempt_payload)
            previous_attempts.append(
                {
                    "attempt": attempt,
                    "sql": attempt_payload["sql"],
                    "explanation": explanation,
                    "error": attempt_payload["error"],
                    "validation_message": expected_validation["message"],
                    "validation_details": expected_validation.get("details", {}),
                }
            )
            if attempt_payload["success"]:
                break

        return _sql_response(
            sql=limited_sql or (sql_attempts[-1]["sql"] if sql_attempts else ""),
            explanation=explanation,
            preview=preview,
            plan=plan,
            metadata=metadata,
            expected_df=expected_df,
            expected_validation=expected_validation,
            sql_attempts=sql_attempts,
            source=source,
            execution_ok=bool(sql_attempts and sql_attempts[-1].get("execution_success")),
        )
    except SQLValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not run read-only query: {type(exc).__name__}") from exc


@router.post("/{connection_id}/explain")
def explain_query(
    connection_id: str,
    database: str = Form(...),
    sql_query: str = Form(...),
    statement_timeout_ms: int = Form(default=5000),
):
    try:
        return explain_read_only_query(connection_id, database, sql_query, statement_timeout_ms)
    except SQLValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not explain query: {type(exc).__name__}") from exc
