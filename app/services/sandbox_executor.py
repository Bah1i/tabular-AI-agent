import json
import pathlib
import shutil
import subprocess
import uuid
import pandas as pd
from app.core.config import settings


class SandboxExecutionError(RuntimeError):
    pass


def execute_code_in_sandbox(code: str, input_df: pd.DataFrame, string_mode: bool = False) -> pd.DataFrame:
    run_id = uuid.uuid4().hex
    workdir = pathlib.Path(settings.sandbox_shared_dir) / run_id
    container_workdir = f"{settings.sandbox_shared_dir}/{run_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        input_path = workdir / "input.csv"
        code_path = workdir / "candidate.py"
        output_path = workdir / "output.csv"
        report_path = workdir / "report.json"

        input_df.to_csv(input_path, index=False)
        code_path.write_text(code, encoding="utf-8")

        command = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", "256m",
            "--cpus", "1",
            "--pids-limit", "128",
            "-e", f"SANDBOX_WORKDIR={container_workdir}",
            "-e", f"SANDBOX_STRING_MODE={'true' if string_mode else 'false'}",
            "-v", f"{settings.sandbox_shared_volume}:{settings.sandbox_shared_dir}:rw",
            settings.sandbox_image,
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=settings.sandbox_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxExecutionError("Sandbox timeout") from exc

        if completed.returncode != 0:
            raise SandboxExecutionError(
                f"Sandbox container failed. stdout={completed.stdout}, stderr={completed.stderr}"
            )

        if not report_path.exists():
            raise SandboxExecutionError("Sandbox report was not created.")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("ok"):
            raise SandboxExecutionError(report.get("error", "Unknown sandbox error"))

        if not output_path.exists():
            raise SandboxExecutionError("Output file was not created.")

        try:
            read_kwargs = {}
            if string_mode:
                read_kwargs.update({"dtype": str, "keep_default_na": False})
            return pd.read_csv(output_path, **read_kwargs)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
