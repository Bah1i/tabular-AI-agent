from pathlib import Path

from app.models.job import TransformJob


def test_create_job_transform_requires_expected(api_client):
    response = api_client.post(
        "/jobs",
        data={"mode": "transform", "instruction": "calculate total"},
        files={"source_file": ("source.csv", "a,b\n1,2\n", "text/csv")},
    )
    assert response.status_code == 400
    assert "Expected file" in response.json()["detail"]


def test_create_job_query_allows_no_expected(api_client):
    response = api_client.post(
        "/jobs",
        data={"mode": "query", "instruction": "top 5"},
        files={"source_file": ("source.csv", "product,total\nA,10\n", "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["job_id"] > 0


def test_preview_endpoint_works(api_client, db_session, tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    job = TransformJob(source_filename="source.csv", source_path=str(source), expected_path=None, mode="query")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    response = api_client.get(f"/jobs/{job.id}/preview/source")

    assert response.status_code == 200
    payload = response.json()
    assert payload["rows_total"] == 2
    assert payload["columns"] == ["a", "b"]


def test_analysis_endpoint_returns_summary(api_client, db_session, tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,x\n, y\n1,x\n", encoding="utf-8")
    job = TransformJob(source_filename="source.csv", source_path=str(source), expected_path=None, mode="query")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    response = api_client.get(f"/jobs/{job.id}/analysis")

    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == 3
    assert "summary" in payload
