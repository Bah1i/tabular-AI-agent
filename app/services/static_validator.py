import ast


FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "requests", "pathlib", "shutil",
    "importlib", "builtins", "inspect", "pickle", "ctypes", "multiprocessing"
}

FORBIDDEN_CALLS = {
    "open", "eval", "exec", "compile", "__import__", "input", "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr"
}

ALLOWED_IMPORTS = {"pandas", "numpy", "re", "math", "datetime", "collections"}


class StaticValidationError(ValueError):
    pass


class FoofahStyleValidationError(StaticValidationError):
    pass


def validate_code_safety(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise StaticValidationError(f"Syntax error: {exc}") from exc

    has_transform = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "transform":
            has_transform = True

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    raise StaticValidationError(f"Forbidden import: {alias.name}")
                if root not in ALLOWED_IMPORTS:
                    raise StaticValidationError(f"Import is not allowed: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                raise StaticValidationError(f"Forbidden import: {node.module}")
            if root not in ALLOWED_IMPORTS:
                raise StaticValidationError(f"Import is not allowed: {node.module}")

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise StaticValidationError(f"Dunder attribute access is forbidden: {node.attr}")

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                raise StaticValidationError(f"Forbidden call: {func.id}")

    if not has_transform:
        raise StaticValidationError("Code must define transform(df).")


def validate_foofah_matrix_style(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise FoofahStyleValidationError(f"Syntax error: {exc}") from exc

    if not _has_matrix_conversion(tree):
        raise FoofahStyleValidationError(
            "FOOFAH code must convert df to a 2D string matrix, for example: "
            "data = df.fillna('').astype(str).values.tolist()."
        )

    for node in ast.walk(tree):
        if _uses_synthetic_column_subscript(node):
            raise FoofahStyleValidationError(
                "FOOFAH code must not access synthetic pandas columns like df['col_4']; "
                "use matrix positions such as data[row][col]."
            )
        if _uses_numeric_conversion(node):
            raise FoofahStyleValidationError(
                "FOOFAH code must preserve cells as strings and must not convert values to numbers."
            )


def _has_matrix_conversion(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        if value is not None and _is_matrix_conversion(value):
            return True
    return False


def _is_matrix_conversion(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tolist"
    ):
        return False

    attrs = _attribute_names(node)
    return (
        "fillna" in attrs
        and "astype" in attrs
        and ("values" in attrs or "to_numpy" in attrs)
    )


def _attribute_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


def _uses_synthetic_column_subscript(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    key = _subscript_key(node.slice)
    if isinstance(key, str):
        return key.startswith("col_")
    if isinstance(key, list):
        return any(isinstance(item, str) and item.startswith("col_") for item in key)
    return False


def _uses_numeric_conversion(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id in {"int", "float", "complex"}:
        return True
    if isinstance(func, ast.Attribute) and func.attr == "to_numeric":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "astype":
        if not node.args:
            return False
        arg = node.args[0]
        if isinstance(arg, ast.Name):
            return arg.id not in {"str", "object"}
        if isinstance(arg, ast.Constant):
            return str(arg.value).lower() not in {"str", "string", "object"}
    return False


def _subscript_key(slice_node: ast.AST):
    if isinstance(slice_node, ast.Constant):
        return slice_node.value
    if isinstance(slice_node, (ast.List, ast.Tuple)):
        values = []
        for item in slice_node.elts:
            if isinstance(item, ast.Constant):
                values.append(item.value)
        return values
    if isinstance(slice_node, ast.Index):  # pragma: no cover - py<3.9 compatibility
        return _subscript_key(slice_node.value)
    return None
