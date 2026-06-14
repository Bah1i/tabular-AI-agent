import hashlib
from pathlib import Path


def compute_file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_text_hash(value: str | None) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def compute_many_text_hash(*values: str | None) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update((value or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
