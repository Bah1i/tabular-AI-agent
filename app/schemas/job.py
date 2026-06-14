from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: int
    status: str


class JobRead(BaseModel):
    id: int
    status: str
    mode: str = "transform"
    prompt_strategy: str = "standard"
    source_filename: str
    user_instruction: str | None = None
    generated_code: str | None = None
    explanation: str | None = None
    validation_report_json: str | None = None
    result_path: str | None = None
    error_message: str | None = None
    source_hash: str | None = None
    expected_hash: str | None = None
    instruction_hash: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    cache_hit_from_job_id: int | None = None
    attempts: int

    class Config:
        from_attributes = True


class RunJobResponse(BaseModel):
    job_id: int
    status: str
    attempts: int
    result_path: str | None = None
    error_message: str | None = None
