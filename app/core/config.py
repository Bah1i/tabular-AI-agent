from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    app_name: str = 'Tabular AI Agent'
    app_env: str = 'dev'
    database_url: str
    redis_url: str = 'redis://redis:6379/0'
    queue_name: str = 'transform_jobs'
    llm_provider: str = 'deepseek'
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    deepseek_api_key: str | None = None
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
    max_prompt_example_rows: int = 25
    preview_rows: int = 100
    large_table_row_threshold: int = 50000
    llm_input_1k_token_price_usd: float = 0.0
    llm_output_1k_token_price_usd: float = 0.0
    max_total_llm_tokens_per_job: int = 0
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    @property
    def effective_llm_api_key(self) -> str:
        value = self.llm_api_key or self.deepseek_api_key
        if not value:
            raise ValueError('LLM_API_KEY or DEEPSEEK_API_KEY must be set.')
        return value

    @property
    def effective_llm_base_url(self) -> str:
        return self.llm_base_url or self.deepseek_base_url

    @property
    def effective_llm_model(self) -> str:
        return self.llm_model or self.deepseek_model
settings = Settings()
