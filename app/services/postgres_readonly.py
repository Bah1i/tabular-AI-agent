from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import psycopg2
from psycopg2 import sql as pg_sql
from psycopg2.extras import RealDictCursor

from app.services.sql_validator import ensure_limit, explain_sql, validate_read_only_select


@dataclass
class PostgresConnection:
    host: str
    port: int
    username: str
    password: str


_CONNECTIONS: dict[str, PostgresConnection] = {}


def register_connection(host: str, port: int, username: str, password: str) -> str:
    connection_id = uuid.uuid4().hex
    _CONNECTIONS[connection_id] = PostgresConnection(host=host, port=int(port), username=username, password=password)
    return connection_id


def _connection_info(connection_id: str) -> PostgresConnection:
    try:
        return _CONNECTIONS[connection_id]
    except KeyError as exc:
        raise ValueError("Unknown PostgreSQL connection id") from exc


def _connect(connection_id: str, database: str):
    info = _connection_info(connection_id)
    return psycopg2.connect(host=info.host, port=info.port, user=info.username, password=info.password, dbname=database)


def list_databases(connection_id: str) -> list[str]:
    info = _connection_info(connection_id)
    with psycopg2.connect(host=info.host, port=info.port, user=info.username, password=info.password, dbname="postgres") as conn:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname")
            return [row[0] for row in cur.fetchall()]


def list_schemas(connection_id: str, database: str) -> list[str]:
    with _connect(connection_id, database) as conn:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT LIKE 'pg_%' AND schema_name <> 'information_schema' ORDER BY schema_name")
            return [row[0] for row in cur.fetchall()]


def list_tables(connection_id: str, database: str, schema: str) -> list[str]:
    with _connect(connection_id, database) as conn:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s AND table_type = 'BASE TABLE' ORDER BY table_name",
                (schema,),
            )
            return [row[0] for row in cur.fetchall()]


def get_schema_metadata(connection_id: str, database: str, schema: str, table: str | None = None) -> dict[str, Any]:
    with _connect(connection_id, database) as conn:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            params: list[Any] = [schema]
            table_filter = ""
            if table:
                table_filter = " AND table_name = %s"
                params.append(table)
            cur.execute(
                """
                SELECT table_name, column_name, data_type, is_nullable, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = %s
                """ + table_filter + " ORDER BY table_name, ordinal_position",
                params,
            )
            rows = cur.fetchall()
            estimate_params: list[Any] = [schema]
            estimate_filter = ""
            if table:
                estimate_filter = " AND c.relname = %s"
                estimate_params.append(table)
            cur.execute(
                """
                SELECT c.relname AS table_name, GREATEST(c.reltuples::bigint, 0) AS row_count_estimate
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = %s
                  AND c.relkind IN ('r', 'p')
                """ + estimate_filter,
                estimate_params,
            )
            row_estimates = {row["table_name"]: row["row_count_estimate"] for row in cur.fetchall()}
    tables: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        tables.setdefault(row["table_name"], []).append(
            {
                "name": row["column_name"],
                "type": row["data_type"],
                "data_type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
                "ordinal_position": row["ordinal_position"],
            }
        )
    return {
        "database": database,
        "schema": schema,
        "tables": [
            {
                "schema": schema,
                "table": name,
                "name": name,
                "row_count_estimate": row_estimates.get(name),
                "columns": columns,
            }
            for name, columns in tables.items()
        ],
        "foreign_keys": [],
    }


def preview_table_rows(
    connection_id: str,
    database: str,
    schema: str,
    table: str,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    with _connect(connection_id, database) as conn:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            table_ident = pg_sql.Identifier(schema, table)
            cur.execute(pg_sql.SQL("SELECT COUNT(*) AS count FROM {}").format(table_ident))
            total_rows = int(cur.fetchone()["count"])
            cur.execute(
                pg_sql.SQL("SELECT * FROM {} ORDER BY ctid LIMIT %s OFFSET %s").format(table_ident),
                (limit, offset),
            )
            rows = [dict(row) for row in cur.fetchall()]
            columns = [desc.name for desc in cur.description] if cur.description else []
    return {
        "database": database,
        "schema": schema,
        "table": table,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "total_rows": total_rows,
        "limit": limit,
        "offset": offset,
    }


def run_read_only_query(connection_id: str, database: str, sql: str, statement_timeout_ms: int = 5000) -> dict[str, Any]:
    limited_sql = ensure_limit(sql)
    with _connect(connection_id, database) as conn:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET statement_timeout = %s", (int(statement_timeout_ms),))
            cur.execute(limited_sql)
            rows = [dict(row) for row in cur.fetchall()]
            columns = [desc.name for desc in cur.description] if cur.description else []
    return {"sql": limited_sql, "columns": columns, "rows": rows, "row_count": len(rows)}


def explain_read_only_query(connection_id: str, database: str, sql: str, statement_timeout_ms: int = 5000) -> dict[str, Any]:
    validate_read_only_select(sql)
    statement = explain_sql(sql)
    with _connect(connection_id, database) as conn:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = %s", (int(statement_timeout_ms),))
            cur.execute(statement)
            rows = [row[0] for row in cur.fetchall()]
    return {"sql": statement, "plan": rows}
