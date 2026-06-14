from sqlalchemy import text

from app.db.session import engine


def run_lightweight_migrations() -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS ala_lens_events (
            id SERIAL PRIMARY KEY,
            job_id INTEGER REFERENCES transform_jobs(id),
            attempt_number INTEGER DEFAULT 1,
            event_type VARCHAR(64) DEFAULT 'get',
            prompt_strategy VARCHAR(64),
            code_hash VARCHAR(64),
            validation_status VARCHAR(64),
            source_model_json TEXT,
            view_model_json TEXT,
            parameter_before_json TEXT,
            delta_json TEXT,
            amendment_json TEXT,
            parameter_after_json TEXT,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ala_lens_typed_deltas (
            id SERIAL PRIMARY KEY,
            event_id INTEGER REFERENCES ala_lens_events(id),
            job_id INTEGER REFERENCES transform_jobs(id),
            attempt_number INTEGER DEFAULT 1,
            event_type VARCHAR(64) DEFAULT 'delta',
            delta_kind VARCHAR(128) DEFAULT 'none',
            raw_error_family VARCHAR(128),
            confidence DOUBLE PRECISION DEFAULT 0,
            putback_policy_name VARCHAR(128),
            putback_target VARCHAR(128),
            amendment_policy VARCHAR(128),
            source_mutation_allowed BOOLEAN DEFAULT false,
            parameter_putback_supported BOOLEAN DEFAULT false,
            restoration_level VARCHAR(128),
            getput_runtime VARCHAR(64),
            putget_runtime VARCHAR(64),
            putput_runtime VARCHAR(64),
            semantic_signature VARCHAR(64),
            typed_delta_json TEXT,
            putback_policy_json TEXT,
            lens_law_checks_json TEXT,
            restoration_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) DEFAULT 'FOOFAH',
            dataset_path VARCHAR(1024) NOT NULL,
            status VARCHAR(32) DEFAULT 'running',
            total_cases INTEGER DEFAULT 0,
            successful_cases INTEGER DEFAULT 0,
            failed_cases INTEGER DEFAULT 0,
            total_latency_seconds DOUBLE PRECISION DEFAULT 0,
            total_estimated_cost_usd DOUBLE PRECISION DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS benchmark_case_results (
            id SERIAL PRIMARY KEY,
            run_id INTEGER REFERENCES benchmark_runs(id),
            job_id INTEGER REFERENCES transform_jobs(id),
            case_name VARCHAR(255),
            input_path VARCHAR(1024) NOT NULL,
            output_path VARCHAR(1024) NOT NULL,
            status VARCHAR(32) DEFAULT 'running',
            success BOOLEAN DEFAULT false,
            attempts INTEGER DEFAULT 0,
            latency_seconds DOUBLE PRECISION DEFAULT 0,
            token_cost_usd DOUBLE PRECISION DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS mode VARCHAR(32)",
        "UPDATE transform_jobs SET mode = 'transform' WHERE mode IS NULL",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS prompt_strategy VARCHAR(64) DEFAULT 'standard'",
        "UPDATE transform_jobs SET prompt_strategy = 'standard' WHERE prompt_strategy IS NULL",
        "ALTER TABLE transform_jobs ALTER COLUMN expected_path DROP NOT NULL",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS explanation TEXT",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS source_profile_json TEXT",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS validation_report_json TEXT",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64)",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS expected_hash VARCHAR(64)",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS instruction_hash VARCHAR(64)",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS model_name VARCHAR(255)",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS prompt_version VARCHAR(64)",
        "ALTER TABLE transform_jobs ADD COLUMN IF NOT EXISTS cache_hit_from_job_id INTEGER",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS llm_calls INTEGER DEFAULT 0",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER DEFAULT 0",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS completion_tokens INTEGER DEFAULT 0",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS total_tokens INTEGER DEFAULT 0",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS estimated_cost_usd DOUBLE PRECISION DEFAULT 0",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS model_name VARCHAR(255)",
        "ALTER TABLE job_metrics ADD COLUMN IF NOT EXISTS cache_hit BOOLEAN DEFAULT false",
        "ALTER TABLE benchmark_case_results ADD COLUMN IF NOT EXISTS dataset_name VARCHAR(255) DEFAULT 'FOOFAH'",
        "ALTER TABLE benchmark_case_results ADD COLUMN IF NOT EXISTS prompt_strategy_used VARCHAR(64)",
        "ALTER TABLE benchmark_case_results ADD COLUMN IF NOT EXISTS fallback_used BOOLEAN DEFAULT false",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS candidate_count INTEGER DEFAULT 1",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS oracle_mode BOOLEAN DEFAULT false",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS use_memory BOOLEAN DEFAULT false",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS benchmark_mode VARCHAR(64) DEFAULT 'strict_honest'",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS memory_enabled BOOLEAN DEFAULT true",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS reuse_case_enabled BOOLEAN DEFAULT false",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS traversal_order VARCHAR(32) DEFAULT 'forward'",
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS benchmark_label VARCHAR(255)",
        "ALTER TABLE benchmark_case_results ADD COLUMN IF NOT EXISTS best_visible_job_id INTEGER",
        "ALTER TABLE benchmark_case_results ADD COLUMN IF NOT EXISTS example_success BOOLEAN DEFAULT false",
        "ALTER TABLE benchmark_case_results ADD COLUMN IF NOT EXISTS generalization_success BOOLEAN",
        "ALTER TABLE benchmark_case_results ADD COLUMN IF NOT EXISTS selected_candidate_index INTEGER",
        "ALTER TABLE benchmark_case_results ADD COLUMN IF NOT EXISTS hidden_judge_message TEXT",
        "ALTER TABLE transformation_memory ADD COLUMN IF NOT EXISTS prompt_version VARCHAR(64)",
        "ALTER TABLE transform_attempts ADD COLUMN IF NOT EXISTS prompt_strategy VARCHAR(64) DEFAULT 'standard'",
        "ALTER TABLE ala_lens_events ADD COLUMN IF NOT EXISTS prompt_strategy VARCHAR(64)",
        "ALTER TABLE ala_lens_events ADD COLUMN IF NOT EXISTS code_hash VARCHAR(64)",
        "ALTER TABLE ala_lens_events ADD COLUMN IF NOT EXISTS validation_status VARCHAR(64)",
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
        if engine.dialect.name == "postgresql":
            legacy_statements = [
                "ALTER TABLE benchmark_case_results DROP COLUMN IF EXISTS is_noisy",
                "ALTER TABLE benchmark_case_results DROP COLUMN IF EXISTS noise_level",
                "ALTER TABLE benchmark_case_results DROP COLUMN IF EXISTS robustness_level",
                "ALTER TABLE benchmark_case_results DROP COLUMN IF EXISTS robustness_profile",
            ]
            for statement in legacy_statements:
                conn.execute(text(statement))
