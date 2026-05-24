from sqlalchemy import text

from app.db.session import engine


def run_lightweight_migrations() -> None:
    statements = [
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS mode VARCHAR(32)",
        "UPDATE transform_jobs SET mode = 'transform' WHERE mode IS NULL",
        "ALTER TABLE transform_jobs ALTER COLUMN expected_path DROP NOT NULL",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS explanation TEXT",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS source_profile_json TEXT",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS validation_report_json TEXT",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS llm_calls INTEGER DEFAULT 0",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER DEFAULT 0",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS completion_tokens INTEGER DEFAULT 0",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS total_tokens INTEGER DEFAULT 0",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS estimated_cost_usd DOUBLE PRECISION DEFAULT 0",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS model_name VARCHAR(255)",
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
