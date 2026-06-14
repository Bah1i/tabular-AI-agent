import re


class SQLValidationError(ValueError):
    pass


def _strip_string_literals(sql: str) -> str:
    result = []
    in_single = False
    in_double = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if in_single:
            if ch == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                i += 2
                continue
            if ch == "'":
                in_single = False
            result.append(" ")
        elif in_double:
            if ch == '"':
                in_double = False
            result.append(" ")
        else:
            if ch == "'":
                in_single = True
                result.append(" ")
            elif ch == '"':
                in_double = True
                result.append(" ")
            else:
                result.append(ch)
        i += 1
    return "".join(result)


def _without_trailing_semicolon(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def validate_read_only_select(sql: str) -> bool:
    cleaned = _without_trailing_semicolon(sql)
    if not cleaned:
        raise SQLValidationError("SQL query is empty.")
    no_literals = _strip_string_literals(cleaned)
    if ";" in no_literals:
        raise SQLValidationError("Only one SQL statement is allowed.")
    lowered = re.sub(r"\s+", " ", no_literals.lower()).strip()
    forbidden = [
        "insert", "update", "delete", "drop", "alter", "truncate", "create", "copy",
        "call", "do", "merge", "grant", "revoke", "vacuum", "analyze", "listen", "notify",
    ]
    for keyword in forbidden:
        if re.search(rf"\b{keyword}\b", lowered):
            raise SQLValidationError(f"Only read-only SELECT queries are allowed; found {keyword}.")
    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        raise SQLValidationError("Only SELECT or WITH ... SELECT queries are allowed.")
    if re.search(r"\bselect\b.+\binto\b", lowered):
        raise SQLValidationError("SELECT INTO is not read-only and is not allowed.")
    if lowered.startswith("with ") and not re.search(r"\)\s*select\b|\bselect\b", lowered):
        raise SQLValidationError("WITH queries must end in SELECT.")
    return True


def ensure_limit(sql: str, limit: int = 1000) -> str:
    validate_read_only_select(sql)
    cleaned = _without_trailing_semicolon(sql)
    no_literals = _strip_string_literals(cleaned).lower()
    if re.search(r"\blimit\s+\d+\b", no_literals):
        return cleaned
    return f"{cleaned} LIMIT {int(limit)}"


def explain_sql(sql: str) -> str:
    validate_read_only_select(sql)
    return f"EXPLAIN {_without_trailing_semicolon(sql)}"
