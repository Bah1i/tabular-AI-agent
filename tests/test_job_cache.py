from sqlalchemy import select

from app.models.benchmark import BenchmarkCaseResult, BenchmarkRun
from app.models.job import JobStatus, TransformJob
from app.models.metric import JobMetric
from app.services.job_cache import apply_cache_hit_if_available, fill_job_cache_keys


def test_cache_hit_copies_successful_job_result(db_session, tmp_path):
    source = tmp_path / "source.csv"
    expected = tmp_path / "expected.csv"
    result = tmp_path / "result.csv"
    source.write_text("product,total\nA,10\n", encoding="utf-8")
    expected.write_text("product,total\nA,10\n", encoding="utf-8")
    result.write_text("product,total\nA,10\n", encoding="utf-8")

    cached_job = TransformJob(
        status=JobStatus.success,
        source_filename="source.csv",
        source_path=str(source),
        expected_path=str(expected),
        user_instruction="keep product and total",
        generated_code="def transform(df): return df",
        explanation="Already solved.",
        result_path=str(result),
        mode="transform",
    )
    fill_job_cache_keys(cached_job)
    db_session.add(cached_job)
    db_session.commit()
    db_session.refresh(cached_job)

    new_job = TransformJob(
        source_filename="source.csv",
        source_path=str(source),
        expected_path=str(expected),
        user_instruction="keep product and total",
        mode="transform",
    )
    fill_job_cache_keys(new_job)
    db_session.add(new_job)
    db_session.commit()
    db_session.refresh(new_job)

    assert apply_cache_hit_if_available(db_session, new_job) is True

    assert new_job.status == JobStatus.success
    assert new_job.cache_hit_from_job_id == cached_job.id
    assert new_job.result_path == cached_job.result_path
    assert new_job.generated_code == cached_job.generated_code
    assert new_job.explanation == cached_job.explanation
    assert new_job.attempts == 0

    metric = db_session.scalars(select(JobMetric).where(JobMetric.job_id == new_job.id)).one()
    assert metric.cache_hit is True
    assert metric.llm_calls == 0


def test_cache_miss_when_instruction_differs(db_session, tmp_path):
    source = tmp_path / "source.csv"
    expected = tmp_path / "expected.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    expected.write_text("a\n1\n", encoding="utf-8")

    cached_job = TransformJob(
        status=JobStatus.success,
        source_filename="source.csv",
        source_path=str(source),
        expected_path=str(expected),
        user_instruction="first instruction",
        result_path=str(tmp_path / "result.csv"),
        mode="transform",
    )
    fill_job_cache_keys(cached_job)
    db_session.add(cached_job)
    db_session.commit()

    new_job = TransformJob(
        source_filename="source.csv",
        source_path=str(source),
        expected_path=str(expected),
        user_instruction="different instruction",
        mode="transform",
    )
    fill_job_cache_keys(new_job)
    db_session.add(new_job)
    db_session.commit()

    assert apply_cache_hit_if_available(db_session, new_job) is False


def test_foofah_exact_cache_requires_benchmark_success(db_session, tmp_path):
    source = tmp_path / "InputTable.csv"
    expected = tmp_path / "OutputTable.csv"
    result = tmp_path / "result.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    expected.write_text("a\n1\n", encoding="utf-8")
    result.write_text("a\n1\n", encoding="utf-8")

    cached_job = TransformJob(
        status=JobStatus.success,
        source_filename="InputTable.csv",
        source_path=str(source),
        expected_path=str(expected),
        user_instruction="Infer the FOOFAH table transformation.",
        generated_code="def transform(df): return df",
        explanation="Visible example passed.",
        result_path=str(result),
        mode="transform",
        prompt_strategy="foofah",
    )
    fill_job_cache_keys(cached_job)
    db_session.add(cached_job)
    db_session.commit()
    db_session.refresh(cached_job)

    visible_only_job = TransformJob(
        source_filename="InputTable.csv",
        source_path=str(source),
        expected_path=str(expected),
        user_instruction="Infer the FOOFAH table transformation.",
        mode="transform",
        prompt_strategy="foofah",
    )
    fill_job_cache_keys(visible_only_job)
    db_session.add(visible_only_job)
    db_session.commit()
    db_session.refresh(visible_only_job)

    assert apply_cache_hit_if_available(db_session, visible_only_job) is False

    run = BenchmarkRun(name="FOOFAH", dataset_path=str(tmp_path), status="success", total_cases=1, successful_cases=1)
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    db_session.add(
        BenchmarkCaseResult(
            run_id=run.id,
            job_id=cached_job.id,
            case_name="exp0_cache_case",
            dataset_name="FOOFAH",
            input_path=str(source),
            output_path=str(expected),
            status="success",
            success=True,
        )
    )
    db_session.commit()

    benchmark_success_job = TransformJob(
        source_filename="InputTable.csv",
        source_path=str(source),
        expected_path=str(expected),
        user_instruction="Infer the FOOFAH table transformation.",
        mode="transform",
        prompt_strategy="foofah",
    )
    fill_job_cache_keys(benchmark_success_job)
    db_session.add(benchmark_success_job)
    db_session.commit()
    db_session.refresh(benchmark_success_job)

    assert apply_cache_hit_if_available(db_session, benchmark_success_job) is True
    assert benchmark_success_job.cache_hit_from_job_id == cached_job.id
