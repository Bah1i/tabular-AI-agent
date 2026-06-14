import pytest

from app.services.sql_validator import SQLValidationError, ensure_limit, explain_sql, validate_read_only_select


@pytest.mark.parametrize("sql", ["select * from users", "WITH x AS (SELECT 1) SELECT * FROM x"])
def test_allows_select_and_with_select(sql):
    assert validate_read_only_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "insert into t values (1)",
        "update t set a=1",
        "delete from t",
        "drop table t",
        "alter table t add column x int",
        "truncate table t",
        "create table t(id int)",
        "copy t to stdout",
        "call do_work()",
        "do $$ begin end $$",
    ],
)
def test_forbids_mutating_or_procedural_sql(sql):
    with pytest.raises(SQLValidationError):
        validate_read_only_select(sql)


def test_rejects_non_select():
    with pytest.raises(SQLValidationError):
        validate_read_only_select("show tables")


def test_rejects_multiple_statements():
    with pytest.raises(SQLValidationError):
        validate_read_only_select("select * from users; select * from orders")


def test_rejects_select_into():
    with pytest.raises(SQLValidationError):
        validate_read_only_select("select * into new_table from users")


def test_with_must_end_with_select():
    with pytest.raises(SQLValidationError):
        validate_read_only_select("with x as (update t set a=1 returning *) select * from x")


def test_forbidden_keywords_inside_string_literals_are_ignored():
    assert validate_read_only_select("select 'drop table users' as text")


def test_ensure_limit_adds_limit_when_missing():
    assert ensure_limit("select * from users").lower().endswith("limit 1000")


def test_ensure_limit_keeps_existing_limit():
    assert ensure_limit("select * from users limit 10").lower().endswith("limit 10")


def test_explain_does_not_use_analyze():
    assert explain_sql("select * from users") == "EXPLAIN select * from users"
