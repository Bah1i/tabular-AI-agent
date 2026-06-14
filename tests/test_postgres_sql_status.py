from app.api.postgres import _sql_lens_payload, _sql_result_status


def test_sql_expected_mismatch_is_not_execution_failure():
    validation = {"ok": False, "message": "Value mismatch examples: []"}

    assert _sql_result_status(True, validation) == "expected_mismatch"

    lens = _sql_lens_payload(
        source="expected_only",
        sql="SELECT COUNT(*) AS total_jobs FROM job_metrics",
        preview={"columns": ["total_jobs"], "rows": [{"total_jobs": 5682}], "row_count": 1},
        expected_validation=validation,
        attempts=[{"attempt": 1, "execution_success": True}],
        execution_ok=True,
    )

    assert lens["status"] == "expected_mismatch"
    assert lens["restoration"]["read_only_execution"] == "ok"
    assert "differs from live database result" in lens["restoration"]["note"]


def test_sql_execution_failure_status_stays_error():
    assert _sql_result_status(False, {"ok": False}) == "execution_failed"
