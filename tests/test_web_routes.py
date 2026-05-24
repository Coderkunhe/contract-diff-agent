"""Integration tests for FastAPI web routes using TestClient."""
import io
import json
import os
import queue
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))

from web.app import app, _JOBS, _JOBS_LOCK

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_jobs():
    """Clear job store before each test."""
    with _JOBS_LOCK:
        _JOBS.clear()


@pytest.fixture
def real_pdf_bytes():
    """Read a small real PDF for upload tests."""
    pdf_path = _PROJECT_ROOT / "docs" / "天猫服务协议2015(2).pdf"
    if pdf_path.exists():
        return pdf_path.read_bytes()
    return None


@pytest.fixture
def test_result_data():
    """Return a minimal but valid result dict."""
    return {
        "meta": {
            "contract_v1": "v1.pdf",
            "contract_v2": "v2.pdf",
            "compared_at": "2026-05-24T00:00:00Z",
            "agent_version": "1.4.0",
            "pipeline": "v04-offline",
            "model": None,
            "token_estimate": {"v1_tokens": 100, "v2_tokens": 120, "total": 220},
        },
        "diff_summary": {
            "total_changes": 2,
            "confirmed": 2,
            "high_risk": 1,
            "medium_risk": 0,
            "low_risk": 1,
            "alignment_coverage": 0.9,
            "llm_enhanced": False,
        },
        "changes": [
            {
                "id": "diff-001",
                "change_type": "modified",
                "brief": "验收期限从7天改为3天",
                "risk_level": "high",
                "risk_note": "时效变短",
                "risk_categories": ["R01"],
                "v1_snippet": "7个工作日内",
                "v2_snippet": "3个工作日内",
                "clause_ref_v1": "第一条",
                "clause_ref_v2": "第一条",
                "source": "algorithm",
                "human_note": None,
            },
            {
                "id": "diff-002",
                "change_type": "added",
                "brief": "新增保密条款",
                "risk_level": "medium",
                "risk_note": "需法务确认",
                "risk_categories": ["R03"],
                "v1_snippet": None,
                "v2_snippet": "保密义务条款全文",
                "clause_ref_v1": None,
                "clause_ref_v2": "第二条",
                "source": "algorithm",
                "human_note": None,
            },
        ],
        "risk_taxonomy_snapshot": {
            "categories_used": ["R01", "R03"],
            "frequency": {"金额条款": 1, "保密义务": 1},
            "high_frequency_alerts": [],
        },
        "unmatched_content": {"v1_only": [], "v2_only": [], "note": ""},
    }


def _setup_job_and_result(job_id: str, result_data: dict, v1_name: str = "v1.pdf", v2_name: str = "v2.pdf"):
    """Create both a job entry and a result file on disk."""
    result_dir = _PROJECT_ROOT / "data" / "jobs"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{job_id}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False)

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id,
            "status": "done",
            "progress": 100,
            "step_label": "完成",
            "message": "比对完成",
            "result_path": str(result_path),
            "created_at": now,
            "v1_filename": v1_name,
            "v2_filename": v2_name,
            "_event_queue": queue.Queue(),
        }
    return str(result_path)


# ── Basic endpoints ───────────────────────────────────────────────

class TestHealth:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "workers" in data


class TestUploadPage:
    def test_root_returns_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ── File upload validation ────────────────────────────────────────

class TestUploadValidation:
    def test_rejects_non_pdf_extension(self):
        resp = client.post(
            "/upload",
            files={
                "v1_file": ("test.txt", io.BytesIO(b"hello world"), "text/plain"),
                "v2_file": ("test.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf"),
            },
        )
        assert resp.status_code == 400
        assert "不是 PDF/DOCX" in resp.json()["error"]

    def test_rejects_empty_file(self):
        resp = client.post(
            "/upload",
            files={
                "v1_file": ("v1.pdf", io.BytesIO(b""), "application/pdf"),
                "v2_file": ("v2.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf"),
            },
        )
        assert resp.status_code == 400

    @pytest.mark.skipif(
        not (_PROJECT_ROOT / "docs" / "天猫服务协议2015(2).pdf").exists(),
        reason="Test PDF not available",
    )
    def test_upload_real_pdf_redirects(self, real_pdf_bytes):
        """Upload two real PDFs and expect a redirect to the job page."""
        # Note: this actually starts the pipeline in a background thread.
        # We mock the executor to avoid real processing.
        with patch("web.app._executor.submit") as mock_submit:
            resp = client.post(
                "/upload",
                files={
                    "v1_file": ("contract_v1.pdf", io.BytesIO(real_pdf_bytes), "application/pdf"),
                    "v2_file": ("contract_v2.pdf", io.BytesIO(real_pdf_bytes), "application/pdf"),
                },
                follow_redirects=False,
            )
            assert resp.status_code == 303
            assert resp.headers["location"].startswith("/job/")
            # Job should be pending
            job_id = resp.headers["location"].split("/")[-1]
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
            assert job is not None
            assert job["status"] in ("queued", "extracting")
            assert job["v1_filename"] == "contract_v1.pdf"

    @pytest.mark.skipif(
        not (_PROJECT_ROOT / "docs" / "天猫服务协议2015(2).pdf").exists(),
        reason="Test PDF not available",
    )
    def test_upload_with_thorough_mode(self, real_pdf_bytes):
        with patch("web.app._executor.submit") as mock_submit:
            resp = client.post(
                "/upload",
                files={
                    "v1_file": ("v1.pdf", io.BytesIO(real_pdf_bytes), "application/pdf"),
                    "v2_file": ("v2.pdf", io.BytesIO(real_pdf_bytes), "application/pdf"),
                },
                data={"thorough": "true"},
                follow_redirects=False,
            )
            assert resp.status_code == 303
            job_id = resp.headers["location"].split("/")[-1]
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
            assert job["thorough"] is True


# ── Job status endpoints ──────────────────────────────────────────

class TestJobStatus:
    def test_job_page_exists(self):
        _setup_job_and_result("abc12345", {})
        resp = client.get("/job/abc12345")
        assert resp.status_code == 200

    def test_job_page_404(self):
        resp = client.get("/job/nonexist")
        assert resp.status_code == 404

    def test_api_job_json(self):
        _setup_job_and_result("abc12345", {})
        resp = client.get("/api/jobs/abc12345")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "abc12345"
        assert data["status"] == "done"
        assert "elapsed_seconds" in data

    def test_api_job_404(self):
        resp = client.get("/api/jobs/nonexist")
        assert resp.status_code == 404


# ── Results endpoints ─────────────────────────────────────────────

class TestResults:
    def test_results_page(self, test_result_data):
        _setup_job_and_result("res001", test_result_data)
        resp = client.get("/results/res001")
        # Should return HTML (either templated or fallback)
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_results_404(self):
        resp = client.get("/results/nonexist")
        assert resp.status_code == 404

    def test_results_json(self, test_result_data):
        _setup_job_and_result("res002", test_result_data)
        resp = client.get("/api/results/res002/json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["diff_summary"]["total_changes"] == 2
        assert len(data["changes"]) == 2

    def test_results_json_404(self):
        resp = client.get("/api/results/nonexist/json")
        assert resp.status_code == 404


# ── Notes endpoint ────────────────────────────────────────────────

class TestNotes:
    def test_update_notes(self, test_result_data):
        _setup_job_and_result("note01", test_result_data)
        resp = client.put(
            "/api/results/note01/notes",
            json={"notes": {"diff-001": "采购部已确认", "diff-002": "法务审核中"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["updated"] == 2

    def test_update_notes_persisted(self, test_result_data):
        """Verify notes are actually written to disk."""
        result_path = _setup_job_and_result("note02", test_result_data)
        resp = client.put(
            "/api/results/note02/notes",
            json={"notes": {"diff-001": "测试备注"}},
        )
        assert resp.status_code == 200

        # Read file back
        with open(result_path, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["changes"][0]["human_note"] == "测试备注"
        assert saved["changes"][1].get("human_note") is None

    def test_update_notes_404(self):
        resp = client.put(
            "/api/results/nonexist/notes",
            json={"notes": {"diff-001": "nope"}},
        )
        assert resp.status_code == 404


# ── PDF generation ────────────────────────────────────────────────

class TestPDF:
    def test_pdf_generation(self, test_result_data):
        _setup_job_and_result("pdf01", test_result_data)
        resp = client.get("/api/results/pdf01/pdf?ids=diff-001,diff-002")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert len(resp.content) > 1000  # Should be a real PDF
        assert resp.content[:4] == b"%PDF"

    def test_pdf_with_all_ids(self, test_result_data):
        """No ids parameter → should include all changes."""
        _setup_job_and_result("pdf02", test_result_data)
        resp = client.get("/api/results/pdf02/pdf")
        assert resp.status_code == 200
        assert len(resp.content) > 500

    def test_pdf_404(self):
        resp = client.get("/api/results/nonexist/pdf")
        assert resp.status_code == 404


# ── SSE streaming ─────────────────────────────────────────────────

class TestSSE:
    def test_sse_stream_exists(self, test_result_data):
        _setup_job_and_result("sse01", test_result_data)
        # Push a done event so the stream terminates cleanly
        with _JOBS_LOCK:
            eq = _JOBS["sse01"]["_event_queue"]
        eq.put_nowait({"event": "done", "data": {"result_url": "/results/sse01"}})
        resp = client.get("/job/sse01/stream")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_sse_404(self):
        resp = client.get("/job/nonexist/stream")
        assert resp.status_code == 404


class TestLearningsPage:
    def test_page_renders_empty(self, tmp_path, monkeypatch):
        """Learnings page renders even with no data."""
        monkeypatch.setattr("src.pipeline.learning._resolve_dir", lambda _: tmp_path)
        (tmp_path / "index.json").write_text(json.dumps({
            "version": "1.0", "total_runs": 0, "runs": [], "global_trends": {},
        }))
        resp = client.get("/learnings")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "暂无进化记录" in resp.text

    def test_page_renders_with_data(self, tmp_path, monkeypatch):
        """Learnings page renders with run data."""
        monkeypatch.setattr("src.pipeline.learning._resolve_dir", lambda _: tmp_path)
        (tmp_path / "index.json").write_text(json.dumps({
            "version": "1.0",
            "total_runs": 1,
            "runs": [{
                "job_id": "abc12345", "timestamp": "2026-05-24T12:00:00Z",
                "pipeline": "v04-llm_enhanced", "model": "deepseek-chat",
                "total_changes": 195, "high_risk": 12, "medium_risk": 45, "low_risk": 138,
                "top_category": {"id": "R01", "name": "交付时效", "count": 25},
                "top_categories": [{"id": "R01", "name": "交付时效", "count": 25}],
                "validation_rejection_rate": 0.05,
                "human_corrections_count": 3,
                "summary": "test summary",
            }],
            "global_trends": {"avg_total_changes": 195, "avg_high_risk_count": 12},
        }))
        resp = client.get("/learnings")
        assert resp.status_code == 200
        assert "abc12345" in resp.text
        assert "deepseek-chat" in resp.text

    def test_page_handles_missing_index(self, tmp_path, monkeypatch):
        """Learnings page works when index.json doesn't exist."""
        monkeypatch.setattr("src.pipeline.learning._resolve_dir", lambda _: tmp_path)
        resp = client.get("/learnings")
        assert resp.status_code == 200
        assert "暂无进化记录" in resp.text


class TestDemo:
    def test_demo_redirects(self):
        """GET /demo redirects to /job/{id} with 303."""
        resp = client.get("/demo", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/job/")

    def test_demo_creates_job(self):
        """GET /demo creates a job in _JOBS."""
        resp = client.get("/demo", follow_redirects=False)
        assert resp.status_code == 303
        job_id = resp.headers["location"].split("/")[-1]
        with _JOBS_LOCK:
            assert job_id in _JOBS
