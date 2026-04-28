from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    app_name: str = 'Tabular AI Agent'
    app_env: str = 'dev'
    database_url: str
    redis_url: str = 'redis://redis:6379/0'
    queue_name: str = 'transform_jobs'
    deepseek_api_key: str
    deepseek_base_url: str = 'https://api.deepseek.com'
    deepseek_model: str = 'deepseek-chat'
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = 'https://cloud.langfuse.com'
    langfuse_enabled: bool = False
    upload_dir: str = '/app/storage/uploads'
    result_dir: str = '/app/storage/results'
    sandbox_image: str = 'tabular-agent-sandbox:latest'
    sandbox_shared_volume: str = 'tabular_agent_sandbox_runs'
    sandbox_shared_dir: str = '/sandbox_runs'
    sandbox_timeout_seconds: int = 15
    max_repair_attempts: int = 3
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
settings = Settings()
