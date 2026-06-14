from pathlib import Path

from sqlalchemy import select

from app.models.ala_lens import AlaLensEvent
from app.models.benchmark import BenchmarkCaseResult, BenchmarkRun
from app.models.job import JobStatus, TransformJob
from app.models.metric import JobMetric
from app.core.config import settings
from app.services.benchmark_runner import (
    BenchmarkRunner,
    _is_not_available_answer,
    benchmark_run_overview,
    benchmark_quality_metrics,
    benchmark_summary,
    discover_foofah_cases,
)
from app.services.job_cache import prompt_cache_version


def test_discover_foofah_cases_reads_input_output_pairs(tmp_path: Path):
    case_dir = tmp_path / "exp0_1_1"
    case_dir.mkdir()
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")

    cases = discover_foofah_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].name == "exp0_1_1"
    assert cases[0].input_path.name == "InputTable.csv"
    assert cases[0].output_path.name == "OutputTable.csv"


def test_discover_foofah_cases_can_select_25_case_block(tmp_path: Path):
    for index in range(30):
        case_dir = tmp_path / f"case_{index:03d}"
        case_dir.mkdir()
        (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
        (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")

    cases = discover_foofah_cases(tmp_path, case_block=2)

    assert len(cases) == 5
    assert cases[0].name == "case_025"
    assert cases[-1].name == "case_029"


def test_foofah_not_available_answer_is_skipped():
    import pandas as pd

    assert _is_not_available_answer(pd.DataFrame([["NOT AVAILABLE, SORRY"]]))
    assert not _is_not_available_answer(pd.DataFrame([["value"]]))


def test_benchmark_runner_with_mock_pipeline(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    for index in range(2):
        case_dir = root / f"exp0_{index}_1"
        case_dir.mkdir(parents=True)
        (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
        (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    def fake_run_transform_job(db, job, max_attempts_override=None):
        job.status = JobStatus.success
        job.attempts = 1
        job.prompt_strategy = job.prompt_strategy or "standard"
        job.generated_code = "def transform(df): return df"
        job.explanation = "Mock benchmark success."
        job.result_path = job.source_path
        db.add(
            JobMetric(
                job_id=job.id,
                success=True,
                attempts=1,
                latency_seconds=0.25,
                total_tokens=12,
                estimated_cost_usd=0.001,
                model_name="mock-model",
            )
        )
        db.add(
            AlaLensEvent(
                job_id=job.id,
                attempt_number=1,
                event_type="get",
                note="Mock ala-lens event.",
            )
        )
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)

    run = BenchmarkRunner(db_session).run_foofah()
    summary = benchmark_summary(db_session, run.id)

    assert summary["total_cases"] == 2
    assert summary["successful_cases"] == 2
    assert summary["failed_cases"] == 0
    assert summary["total_estimated_cost_usd"] == 0.002
    assert "law_check_summary" in summary
    assert "restoration_success_rate_by_delta_family" in summary
    assert summary["explicit_putback_mode"]["source_mutation"] == "forbidden_by_default"
    assert all(case["job_id"] for case in summary["cases"])
    assert all(case["prompt_strategy_used"] == "foofah" for case in summary["cases"])

    results = list(db_session.scalars(select(BenchmarkCaseResult).where(BenchmarkCaseResult.run_id == run.id)).all())
    assert len(results) == 2
    assert all(item.success for item in results)

    lens_events = list(db_session.scalars(select(AlaLensEvent)).all())
    assert len(lens_events) == 2


def test_foofah_benchmark_uses_only_foofah_prompt(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    def fake_run_transform_job(db, job, max_attempts_override=None):
        job.attempts = 1
        job.status = JobStatus.success if job.prompt_strategy == "foofah" else JobStatus.failed
        job.result_path = job.source_path if job.status == JobStatus.success else None
        job.error_message = None if job.status == JobStatus.success else "standard prompt failed"
        db.add(JobMetric(job_id=job.id, success=job.status == JobStatus.success, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)

    run = BenchmarkRunner(db_session).run_foofah()
    summary = benchmark_summary(db_session, run.id)

    assert summary["successful_cases"] == 1
    assert summary["cases"][0]["prompt_strategy_used"] == "foofah"
    assert summary["cases"][0]["fallback_used"] is False


def test_foofah_strict_honest_summary_exposes_clean_mode(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    seen_instructions = []

    def fake_run_transform_job(db, job, max_attempts_override=None):
        seen_instructions.append(job.user_instruction)
        job.status = JobStatus.success
        job.attempts = 1
        job.result_path = job.source_path
        job.generated_code = "def transform(df):\n    return df"
        db.add(JobMetric(job_id=job.id, success=True, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)

    run = BenchmarkRunner(db_session).run_foofah(memory_enabled=True, reuse_case_enabled=False)
    summary = benchmark_summary(db_session, run.id)
    overview = benchmark_run_overview(db_session, run.id)

    assert summary["benchmark_mode"] == "strict_honest"
    assert summary["mode_label"] == "honest"
    assert summary["memory_enabled"] is True
    assert summary["reuse_case_enabled"] is False
    assert summary["traversal_order"] == "forward"
    assert overview["benchmark_mode"] == "strict_honest"
    assert '"benchmark_mode": "strict_honest"' in seen_instructions[0]
    assert '"hidden_feedback_allowed": false' in seen_instructions[0]
    assert "benchmark_generalization_feedback" not in seen_instructions[0]
    assert "test_answer_grid" not in seen_instructions[0]


def test_foofah_strict_honest_does_not_call_reuse_case_path(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    def fail_reuse(*args, **kwargs):
        raise AssertionError("strict_honest must not call reuse case path")

    def fake_run_transform_job(db, job, max_attempts_override=None):
        job.status = JobStatus.success
        job.attempts = 1
        job.result_path = job.source_path
        job.generated_code = "def transform(df):\n    return df"
        db.add(JobMetric(job_id=job.id, success=True, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr(BenchmarkRunner, "_try_reuse_successful_foofah_solution", fail_reuse)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)

    run = BenchmarkRunner(db_session).run_foofah(reuse_case_enabled=False)
    summary = benchmark_summary(db_session, run.id)

    assert summary["successful_cases"] == 1
    assert summary["benchmark_mode"] == "strict_honest"
    assert summary["reuse_case_enabled"] is False


def test_foofah_candidate_count_alone_keeps_strict_honest_mode(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    def fail_reuse(*args, **kwargs):
        raise AssertionError("candidate_count must not enable reuse_case path")

    def fake_run_transform_job(db, job, max_attempts_override=None):
        job.status = JobStatus.success
        job.attempts = 1
        job.result_path = job.source_path
        job.generated_code = "def transform(df):\n    return df"
        db.add(JobMetric(job_id=job.id, success=True, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr(BenchmarkRunner, "_try_reuse_successful_foofah_solution", fail_reuse)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)

    run = BenchmarkRunner(db_session).run_foofah(candidate_count=3)
    summary = benchmark_summary(db_session, run.id)
    overview = benchmark_run_overview(db_session, run.id)

    assert summary["candidate_count"] == 3
    assert summary["benchmark_mode"] == "strict_honest"
    assert summary["mode_label"] == "honest"
    assert summary["memory_enabled"] is True
    assert summary["reuse_case_enabled"] is False
    assert summary["traversal_order"] == "forward"
    assert overview["benchmark_mode"] == "strict_honest"
    assert overview["reuse_case_enabled"] is False


def test_foofah_middle_honest_uses_reverse_traversal_without_hidden_feedback(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    for name in ["exp0_1_1", "exp0_2_1"]:
        case_dir = root / name
        case_dir.mkdir(parents=True)
        (case_dir / "InputTable.csv").write_text(f"{name}\n", encoding="utf-8")
        (case_dir / "OutputTable.csv").write_text(f"{name}\n", encoding="utf-8")
        (case_dir / "TestingTable.csv").write_text("hidden\n", encoding="utf-8")
        (case_dir / "TestAnswer.csv").write_text("answer\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    seen_cases = []
    seen_instructions = []

    def fake_run_transform_job(db, job, max_attempts_override=None):
        seen_cases.append(Path(job.source_path).parent.name)
        seen_instructions.append(job.user_instruction)
        job.status = JobStatus.success
        job.attempts = 1
        job.result_path = job.source_path
        job.generated_code = "def transform(df):\n    return df"
        db.add(JobMetric(job_id=job.id, success=True, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr(BenchmarkRunner, "_try_reuse_successful_foofah_solution", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)
    monkeypatch.setattr(BenchmarkRunner, "_validate_foofah_hidden_test", lambda self, job, case: "hidden mismatch")

    run = BenchmarkRunner(db_session).run_foofah(candidate_count=2, reuse_case_enabled=True)
    summary = benchmark_summary(db_session, run.id)
    overview = benchmark_run_overview(db_session, run.id)

    assert seen_cases == ["exp0_2_1", "exp0_2_1", "exp0_1_1", "exp0_1_1"]
    assert summary["benchmark_mode"] == "middle_honest"
    assert summary["mode_label"] == "little tricky / middle honest"
    assert summary["memory_enabled"] is True
    assert summary["reuse_case_enabled"] is True
    assert summary["traversal_order"] == "reverse"
    assert overview["benchmark_mode"] == "middle_honest"
    assert overview["mode_label"] == "little tricky / middle honest"
    assert overview["reuse_case_enabled"] is True
    assert overview["traversal_order"] == "reverse"
    for instruction in seen_instructions:
        assert '"benchmark_mode": "middle_honest"' in instruction
        assert '"hidden_feedback_allowed": false' in instruction
        assert "benchmark_generalization_feedback" not in instruction
        assert "test_answer_grid" not in instruction
        assert "hidden mismatch" not in instruction


def test_benchmark_ui_declares_mode_row_classes():
    template = Path("app/templates/benchmarks.html").read_text(encoding="utf-8")
    styles = Path("app/static/styles.css").read_text(encoding="utf-8")

    assert "row-middle-honest" in template
    assert "row-oracle" in template
    assert "Cache hits" in template
    assert "mode-preview" in template
    assert "reuseCaseInput.checked = false" in template
    assert "memoryInput.checked = true" in template
    assert ".row-middle-honest td" in styles
    assert "#fff9d8" in styles
    assert ".row-oracle td" in styles
    assert "#fff1f1" in styles


def test_foofah_hidden_mismatch_does_not_feed_answer_back_to_repair(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "TestingTable.csv").write_text("b\n2\n", encoding="utf-8")
    (case_dir / "TestAnswer.csv").write_text("c\n3\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    seen_budgets = []

    def fake_run_transform_job(db, job, max_attempts_override=None):
        seen_budgets.append(max_attempts_override)
        job.attempts = min(2, max_attempts_override or 2)
        job.status = JobStatus.success
        job.result_path = job.source_path
        job.generated_code = "def transform(df):\n    return df"
        db.add(JobMetric(job_id=job.id, success=True, attempts=job.attempts, latency_seconds=0.1))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)
    monkeypatch.setattr(BenchmarkRunner, "_validate_foofah_hidden_test", lambda self, job, case: "hidden mismatch")

    run = BenchmarkRunner(db_session).run_foofah()
    summary = benchmark_summary(db_session, run.id)

    assert summary["successful_cases"] == 0
    assert summary["cases"][0]["attempts"] == 2
    assert seen_budgets == [5]


def test_foofah_expensive_candidate_mode_runs_next_candidate_after_visible_failure(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "TestingTable.csv").write_text("b\n2\n", encoding="utf-8")
    (case_dir / "TestAnswer.csv").write_text("b\n2\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    seen_instructions = []
    seen_budgets = []

    def fake_run_transform_job(db, job, max_attempts_override=None):
        seen_instructions.append(job.user_instruction)
        seen_budgets.append(max_attempts_override)
        job.attempts = 1
        job.status = JobStatus.failed if len(seen_instructions) == 1 else JobStatus.success
        job.result_path = job.source_path
        job.generated_code = f"def transform(df):\n    return df  # candidate {len(seen_instructions)}"
        db.add(JobMetric(job_id=job.id, success=True, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)

    run = BenchmarkRunner(db_session).run_foofah(candidate_count=2)
    summary = benchmark_summary(db_session, run.id)

    assert summary["successful_cases"] == 1
    assert summary["cases"][0]["attempts"] == 2
    assert summary["cases"][0]["fallback_used"] is True
    assert seen_budgets == [5, 5]
    assert '"candidate_index": 1' in seen_instructions[0]
    assert '"candidate_index": 2' in seen_instructions[1]


def test_foofah_honest_mode_does_not_use_hidden_answer_for_repair(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "TestingTable.csv").write_text("b\n2\n", encoding="utf-8")
    (case_dir / "TestAnswer.csv").write_text("c\n3\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    seen_instructions = []

    def fake_run_transform_job(db, job, max_attempts_override=None):
        seen_instructions.append(job.user_instruction)
        job.attempts = 1
        job.status = JobStatus.success
        job.result_path = job.source_path
        job.generated_code = "def transform(df):\n    return df"
        db.add(JobMetric(job_id=job.id, success=True, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)
    monkeypatch.setattr(BenchmarkRunner, "_validate_foofah_hidden_test", lambda self, job, case: "hidden mismatch")

    run = BenchmarkRunner(db_session).run_foofah(candidate_count=3)
    summary = benchmark_summary(db_session, run.id)

    assert summary["successful_cases"] == 0
    assert len(seen_instructions) == 3
    assert summary["cases"][0]["example_success"] is True
    assert summary["cases"][0]["generalization_success"] is False
    assert summary["cases"][0]["selected_candidate_index"] == 3
    for instruction in seen_instructions:
        assert "benchmark_generalization_feedback" not in instruction
        assert "test_answer_grid" not in instruction
        assert "hidden mismatch" not in instruction


def test_foofah_keeps_last_visible_success_artifact_when_hidden_fails(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "TestingTable.csv").write_text("b\n2\n", encoding="utf-8")
    (case_dir / "TestAnswer.csv").write_text("c\n3\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    seen_jobs = []
    seen_instructions = []

    def fake_run_transform_job(db, job, max_attempts_override=None):
        seen_jobs.append(job.id)
        seen_instructions.append(job.user_instruction)
        job.attempts = 1
        job.status = JobStatus.failed if len(seen_jobs) == 2 else JobStatus.success
        job.result_path = job.source_path if job.status == JobStatus.success else None
        job.generated_code = f"def transform(df):\n    return df  # candidate {len(seen_jobs)}"
        db.add(JobMetric(job_id=job.id, success=job.status == JobStatus.success, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)
    monkeypatch.setattr(BenchmarkRunner, "_validate_foofah_hidden_test", lambda self, job, case: "hidden mismatch")

    run = BenchmarkRunner(db_session).run_foofah(candidate_count=3)
    summary = benchmark_summary(db_session, run.id)
    case_summary = summary["cases"][0]

    assert summary["successful_cases"] == 0
    assert case_summary["benchmark_success"] is False
    assert case_summary["example_success"] is True
    assert case_summary["generalization_success"] is False
    assert case_summary["selected_candidate_index"] == 3
    assert case_summary["job_id"] == seen_jobs[2]
    assert case_summary["best_visible_job_id"] == seen_jobs[2]
    for instruction in seen_instructions:
        assert "benchmark_generalization_feedback" not in instruction
        assert "test_answer_grid" not in instruction
        assert "hidden mismatch" not in instruction


def test_foofah_candidate_two_hidden_pass_selects_candidate_two_artifact(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "TestingTable.csv").write_text("b\n2\n", encoding="utf-8")
    (case_dir / "TestAnswer.csv").write_text("b\n2\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    seen_jobs = []

    def fake_run_transform_job(db, job, max_attempts_override=None):
        seen_jobs.append(job.id)
        job.attempts = 1
        job.status = JobStatus.success
        job.result_path = job.source_path
        job.generated_code = f"def transform(df):\n    return df  # candidate {len(seen_jobs)}"
        db.add(JobMetric(job_id=job.id, success=True, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    hidden_results = iter(["hidden mismatch", None])

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)
    monkeypatch.setattr(BenchmarkRunner, "_validate_foofah_hidden_test", lambda self, job, case: next(hidden_results))

    run = BenchmarkRunner(db_session).run_foofah(candidate_count=3)
    summary = benchmark_summary(db_session, run.id)
    case_summary = summary["cases"][0]

    assert summary["successful_cases"] == 1
    assert len(seen_jobs) == 2
    assert case_summary["benchmark_success"] is True
    assert case_summary["example_success"] is True
    assert case_summary["generalization_success"] is True
    assert case_summary["selected_candidate_index"] == 2
    assert case_summary["fallback_used"] is True
    assert case_summary["job_id"] == seen_jobs[1]
    assert case_summary["best_visible_job_id"] == seen_jobs[1]


def test_foofah_oracle_mode_feeds_hidden_answer_to_next_attempt(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "TestingTable.csv").write_text("b\n2\n", encoding="utf-8")
    (case_dir / "TestAnswer.csv").write_text("c\n3\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    seen_instructions = []

    def fake_run_transform_job(db, job, max_attempts_override=None):
        seen_instructions.append(job.user_instruction)
        job.attempts = 1
        job.status = JobStatus.success
        job.result_path = job.source_path
        job.generated_code = "def transform(df):\n    return df"
        db.add(JobMetric(job_id=job.id, success=True, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    hidden_results = iter(["hidden mismatch", None])

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)
    monkeypatch.setattr(BenchmarkRunner, "_validate_foofah_hidden_test", lambda self, job, case: next(hidden_results))

    run = BenchmarkRunner(db_session).run_foofah(candidate_count=1, oracle_mode=True)
    summary = benchmark_summary(db_session, run.id)

    assert summary["successful_cases"] == 1
    assert len(seen_instructions) == 2
    assert '"foofah_cache_mode": "oracle"' in seen_instructions[0]
    assert "benchmark_generalization_feedback" in seen_instructions[1]
    assert "testing_input_grid" in seen_instructions[1]
    assert "testing_table_grid" in seen_instructions[1]
    assert "test_answer_grid" in seen_instructions[1]


def test_foofah_reuses_successful_case_for_same_prompt_version(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")

    cached_job = TransformJob(
        source_filename="InputTable.csv",
        source_path=str(case_dir / "InputTable.csv"),
        expected_path=str(case_dir / "OutputTable.csv"),
        user_instruction="cached",
        mode="transform",
        prompt_strategy="foofah",
        status=JobStatus.success,
        generated_code="def transform(df):\n    return df",
        model_name=settings.effective_llm_model,
        prompt_version=prompt_cache_version("foofah", cache_mode="middle_honest"),
    )
    db_session.add(cached_job)
    db_session.commit()
    db_session.refresh(cached_job)
    old_run = BenchmarkRun(
        name="FOOFAH",
        dataset_path=str(root),
        status="failed",
        total_cases=1,
        successful_cases=0,
        failed_cases=1,
        benchmark_mode="middle_honest",
        memory_enabled=True,
        reuse_case_enabled=True,
        traversal_order="reverse",
    )
    db_session.add(old_run)
    db_session.commit()
    db_session.refresh(old_run)
    db_session.add(
        BenchmarkCaseResult(
            run_id=old_run.id,
            job_id=cached_job.id,
            case_name="exp0_1_1",
            dataset_name="FOOFAH",
            input_path=str(case_dir / "InputTable.csv"),
            output_path=str(case_dir / "OutputTable.csv"),
            status="failed",
            success=False,
            example_success=True,
            generalization_success=False,
            error_message="hidden mismatch",
        )
    )
    db_session.commit()

    def fake_find_foofah_root():
        return root

    def fake_execute(code, df, string_mode=False):
        return df.copy()

    def fail_run_transform_job(*args, **kwargs):
        raise AssertionError("LLM pipeline should not run for cached successful FOOFAH case")

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.execute_code_in_sandbox", fake_execute)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fail_run_transform_job)

    run = BenchmarkRunner(db_session).run_foofah(candidate_count=3, reuse_successful=True)
    summary = benchmark_summary(db_session, run.id)

    assert summary["successful_cases"] == 1
    assert summary["benchmark_mode"] == "middle_honest"
    assert summary["cases"][0]["attempts"] == 0
    assert summary["cases"][0]["total_tokens"] == 0
    assert summary["cases"][0]["cache_hit"] is True


def test_foofah_reuse_prefers_current_run_family_history(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    for name in ["exp0_7_1", "exp0_7_2"]:
        case_dir = root / name
        case_dir.mkdir(parents=True)
        (case_dir / "InputTable.csv").write_text(f"{name}\n", encoding="utf-8")
        (case_dir / "OutputTable.csv").write_text(f"{name}\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    llm_calls = []

    def fake_run_transform_job(db, job, max_attempts_override=None):
        llm_calls.append(Path(job.source_path).parent.name)
        job.status = JobStatus.success
        job.attempts = 1
        job.result_path = job.source_path
        job.generated_code = "def transform(df):\n    return df"
        db.add(JobMetric(job_id=job.id, success=True, attempts=1, latency_seconds=0.1, total_tokens=10))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)

    run = BenchmarkRunner(db_session).run_foofah(candidate_count=1, reuse_case_enabled=True)
    summary = benchmark_summary(db_session, run.id)

    assert llm_calls == ["exp0_7_2"]
    assert summary["benchmark_mode"] == "middle_honest"
    assert [case["case_name"] for case in summary["cases"]] == ["exp0_7_2", "exp0_7_1"]
    assert summary["cases"][0]["cache_hit"] is False
    assert summary["cases"][1]["cache_hit"] is True
    assert summary["cases"][1]["selected_candidate_index"] == 0


def test_foofah_cache_does_not_cross_honest_and_oracle_modes(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a\n1\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a\n1\n", encoding="utf-8")

    cached_job = TransformJob(
        source_filename="InputTable.csv",
        source_path=str(case_dir / "InputTable.csv"),
        expected_path=str(case_dir / "OutputTable.csv"),
        user_instruction="cached oracle",
        mode="transform",
        prompt_strategy="foofah",
        status=JobStatus.success,
        generated_code="def transform(df):\n    return df",
        model_name=settings.effective_llm_model,
        prompt_version=prompt_cache_version("foofah", cache_mode="oracle"),
    )
    db_session.add(cached_job)
    db_session.commit()
    db_session.refresh(cached_job)
    old_run = BenchmarkRun(name="FOOFAH", dataset_path=str(root), status="success", total_cases=1, successful_cases=1)
    db_session.add(old_run)
    db_session.commit()
    db_session.refresh(old_run)
    db_session.add(
        BenchmarkCaseResult(
            run_id=old_run.id,
            job_id=cached_job.id,
            case_name="exp0_1_1",
            dataset_name="FOOFAH",
            input_path=str(case_dir / "InputTable.csv"),
            output_path=str(case_dir / "OutputTable.csv"),
            status="success",
            success=True,
            example_success=True,
        )
    )
    db_session.commit()

    def fake_find_foofah_root():
        return root

    calls = {"count": 0}

    def fake_run_transform_job(db, job, max_attempts_override=None):
        calls["count"] += 1
        job.status = JobStatus.success
        job.attempts = 1
        job.result_path = job.source_path
        job.generated_code = "def transform(df):\n    return df"
        db.add(JobMetric(job_id=job.id, success=True, attempts=1, latency_seconds=0.1))
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)

    run = BenchmarkRunner(db_session).run_foofah(reuse_successful=True, oracle_mode=False)
    summary = benchmark_summary(db_session, run.id)

    assert calls["count"] == 1
    assert summary["cases"][0]["cache_hit"] is False


def test_benchmark_quality_metrics_reports_reliability(monkeypatch, db_session, tmp_path: Path):
    root = tmp_path / "foofah-csv-with-comma"
    case_dir = root / "exp0_1_1"
    case_dir.mkdir(parents=True)
    (case_dir / "InputTable.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    (case_dir / "OutputTable.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    def fake_find_foofah_root():
        return root

    def fake_run_transform_job(db, job, max_attempts_override=None):
        job.status = JobStatus.success
        job.attempts = 1
        job.result_path = job.source_path
        db.add(
            JobMetric(
                job_id=job.id,
                success=True,
                attempts=job.attempts,
                latency_seconds=0.5,
                total_tokens=20,
                estimated_cost_usd=0.002,
                model_name="mock-model",
            )
        )
        db.commit()
        return job

    monkeypatch.setattr("app.services.benchmark_runner.find_foofah_root", fake_find_foofah_root)
    monkeypatch.setattr("app.services.benchmark_runner.run_transform_job", fake_run_transform_job)

    run = BenchmarkRunner(db_session).run_foofah()
    metrics = benchmark_quality_metrics(db_session, run.id)

    assert metrics["reliability"]["success_rate"] == 1.0
    assert metrics["reliability"]["variance_of_attempts"] is not None
    assert metrics["reliability"]["failure_rate_by_dataset"]["FOOFAH"] == 0.0
