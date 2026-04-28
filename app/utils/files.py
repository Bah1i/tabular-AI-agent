import os
import uuid
import pathlib
from fastapi import UploadFile


def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


async def save_upload(upload: UploadFile, directory: str) -> str:
    ensure_dir(directory)

    suffix = pathlib.Path(upload.filename or "").suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    target_path = pathlib.Path(directory) / safe_name

    content = await upload.read()
    target_path.write_bytes(content)

    return str(target_path)