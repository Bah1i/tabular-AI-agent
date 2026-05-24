import ast


FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "requests", "pathlib", "shutil",
    "importlib", "builtins", "inspect", "pickle", "ctypes", "multiprocessing"
}

FORBIDDEN_CALLS = {
    "open", "eval", "exec", "compile", "__import__", "input", "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr"
}

ALLOWED_IMPORTS = {"pandas", "numpy", "re", "math", "datetime"}


class StaticValidationError(ValueError):
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
