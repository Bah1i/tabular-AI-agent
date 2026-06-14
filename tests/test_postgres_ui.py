from pathlib import Path


def test_postgres_page_has_paginated_db_and_expected_previews():
    template = Path("app/templates/postgres.html").read_text(encoding="utf-8")

    assert "Data preview" in template
    assert "db-preview-table" in template
    assert "expected-preview-table" in template
    assert "loadDbPreview" in template
    assert "loadExpectedPreview" in template
    assert "table.table || table.name" in template
    assert "col.type || col.data_type" in template
    assert "/postgres/expected-preview" in template
    assert "/table-preview?" in template
    assert "query-status" in template
    assert "sql-lens-summary" in template
    assert "generated-sql-output" in template
    assert "renderSqlLens" in template
    assert "renderGeneratedSql" in template
    assert "SQL выполнился, но expected-файл не совпал" in template
