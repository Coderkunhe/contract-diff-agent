"""FastAPI web app for the Contract Diff Agent.

Usage:
    python -m web.app
    make web
"""

import argparse
import atexit
import io
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import AppConfig
from src.pdf_extractor import extract_contract
from src.main import _run_v03

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(title="合同差异比对工具", version="0.4.0")
templates_dir = Path(__file__).resolve().parent / "templates"

# ── Config ────────────────────────────────────────────────────────
config = AppConfig.from_env()
_LLM_ENABLED = bool(config.api_key)

# ── Thread pool ───────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="diffworker")


def _shutdown_executor():
    """Shutdown thread pool gracefully on exit."""
    _executor.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_executor)

# ── Job store ─────────────────────────────────────────────────────
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 20

_JOB_STATUS_STEPS = {
    "queued": (0, "排队中"),
    "extracting": (5, "提取 PDF 文本"),
    "tree_building": (10, "构建条款树"),
    "aligning": (20, "对齐条款"),
    "identifying": (30, "LLM 识别差异"),
    "validating": (55, "校验变化"),
    "classifying": (75, "风险分类"),
    "done": (100, "完成"),
    "error": (0, "出错"),
}


def _lr_evict():
    """Remove oldest job if we have too many."""
    with _JOBS_LOCK:
        if len(_JOBS) >= _MAX_JOBS:
            oldest = min(_JOBS.keys(), key=lambda k: _JOBS[k].get("created_at", datetime.min))
            del _JOBS[oldest]


# ── Progress capture ─────────────────────────────────────────────
class JobProgressIO(io.StringIO):
    """Capture pipeline stdout and update job progress in real-time."""

    def __init__(self, job_id: str, real_stdout):
        super().__init__()
        self._job_id = job_id
        self._real = real_stdout
        self._line_buf = ""

    def write(self, s: str):
        self._real.write(s)
        self._real.flush()
        super().write(s)
        self._line_buf += s
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            self._parse_line(line.strip())

    def flush(self):
        self._real.flush()

    def _parse_line(self, line: str):
        if not line:
            return
        jid = self._job_id
        now = datetime.now(timezone.utc)
        # Stage detection — update job AND push SSE event
        stage_info = None
        if "①" in line or "构建条款树" in line:
            stage_info = ("tree_building", 10, "构建条款树")
        elif "②" in line or "条款对齐" in line:
            stage_info = ("aligning", 20, "条款对齐")
        elif "③" in line or "差异识别" in line:
            stage_info = ("identifying", 30, "LLM 逐条比对差异")
        elif "④" in line or "校验" in line:
            stage_info = ("validating", 55, "校验差异")
        elif "⑤" in line or "风险分类" in line:
            stage_info = ("classifying", 75, "风险分类")

        if stage_info:
            status, progress, label = stage_info
            _update_job_inner(jid, status=status, progress=progress, message=line, timestamp=now)
            _send_event(jid, "progress", {"status": status, "step": label, "progress": progress, "message": line})

        # Sub-stage progress
        m = re.search(r"校验进度:\s*(\d+)/(\d+)", line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            pct = 55 + int(20 * cur / total) if total else 55
            _update_job_inner(jid, progress=pct, message=line, timestamp=now)
        m = re.search(r"batch\s+(\d+)/(\d+)", line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            pct = 75 + int(25 * cur / total) if total else 75
            _update_job_inner(jid, progress=pct, message=line, timestamp=now)
        # Per-clause progress: "[N/M] title... -> X changes"
        m = re.search(r"\[(\d+)/(\d+)\]\s", line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            pct = 30 + int(25 * cur / total) if total else 30
            _update_job_inner(jid, progress=pct, message=line, timestamp=now)
        # Accumulate log
        _append_log_inner(jid, line)
        _update_job_inner(jid, message=line, timestamp=now)


def _update_job_inner(job_id: str, **kwargs):
    with _JOBS_LOCK:
        if job_id in _JOBS:
            j = _JOBS[job_id]
            for k, v in kwargs.items():
                if k == "timestamp" or k == "progress" or k == "message" or k == "status":
                    pass  # handled below
            if "status" in kwargs:
                j["status"] = kwargs["status"]
                if kwargs["status"] in _JOB_STATUS_STEPS:
                    pct, label = _JOB_STATUS_STEPS[kwargs["status"]]
                    if "progress" not in kwargs:
                        j["progress"] = pct
                    j["step_label"] = label
            if "progress" in kwargs:
                j["progress"] = kwargs["progress"]
            if "message" in kwargs:
                j["message"] = kwargs["message"]


def _append_log_inner(job_id: str, line: str):
    with _JOBS_LOCK:
        if job_id in _JOBS:
            logs = _JOBS[job_id].setdefault("log_lines", [])
            logs.append(line)
            if len(logs) > 50:
                logs[:] = logs[-50:]


def _send_event(job_id: str, event_type: str, data: dict):
    """Push an SSE event to the job's event queue."""
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
    if j and j.get("_event_queue"):
        try:
            j["_event_queue"].put_nowait({"event": event_type, "data": data})
        except queue.Full:
            pass


# ── Pipeline worker ──────────────────────────────────────────────
def _run_pipeline(job_id: str, v1_path: str, v2_path: str, keep_english: bool):
    """Run pipeline in background thread, streaming results via SSE."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        return

    # Callback for streaming progress
    def _on_progress(status, progress, step, msg):
        _send_event(job_id, "progress", {
            "status": status, "progress": progress,
            "step": step, "message": msg,
        })

    # Callback for streaming changes
    def _on_change(change: dict):
        _send_event(job_id, "change", {
            "id": change.get("id", ""),
            "change_type": change.get("change_type", "modified"),
            "brief": change.get("brief", ""),
            "risk_level": change.get("risk_level", "medium"),
            "risk_note": change.get("risk_note", ""),
            "risk_categories": change.get("risk_categories", []),
            "v1_snippet": change.get("v1_snippet", ""),
            "v2_snippet": change.get("v2_snippet", ""),
            "clause_ref_v2": change.get("clause_ref_v2") or change.get("clause_ref_v1", ""),
            "attention_for": change.get("attention_for"),
            "is_favorable": change.get("is_favorable"),
        })

    try:
        now = datetime.now(timezone.utc)
        _update_job_inner(job_id, status="extracting", progress=3,
                          message="正在提取 V1 PDF 文本...", timestamp=now)
        _send_event(job_id, "progress", {"status": "extracting", "step": "解析 V1 文档",
                      "progress": 3, "message": "正在提取 V1 合同文本..."})

        v1 = extract_contract(v1_path, keep_english=keep_english)
        _send_event(job_id, "progress", {"status": "extracting", "step": "解析 V2 文档",
                      "progress": 6, "message": f"V1: {v1.total_pages} 页, 正在提取 V2..."})

        v2 = extract_contract(v2_path, keep_english=keep_english)
        _send_event(job_id, "progress", {"status": "extracting", "step": "文档解析完成",
                      "progress": 9, "message": f"V1: {v1.total_pages} 页, V2: {v2.total_pages} 页"})

        fake_args = argparse.Namespace(validate=False)

        _send_event(job_id, "progress", {"status": "tree_building", "step": "构建条款树",
                      "progress": 12, "message": "正在解析合同条款结构..."})

        # Capture stdout for progress tracking
        tracker = JobProgressIO(job_id, sys.stdout)
        old_stdout = sys.stdout
        sys.stdout = tracker

        try:
            result = _run_v03(
                fake_args, v1, v2,
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.model,
                on_change=_on_change,
                on_progress=_on_progress,
            )
        finally:
            sys.stdout = old_stdout

        # Send summary
        s = result.get("diff_summary", {})
        _send_event(job_id, "summary", {
            "total_changes": s.get("total_changes", 0),
            "high_risk": s.get("high_risk", 0),
            "medium_risk": s.get("medium_risk", 0),
            "low_risk": s.get("low_risk", 0),
        })

        # Save result
        result_dir = _PROJECT_ROOT / "data" / "jobs"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{job_id}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        now = datetime.now(timezone.utc)
        _update_job_inner(job_id, status="done", progress=100,
                          result_path=str(result_path),
                          message="比对完成", timestamp=now)
        _send_event(job_id, "done", {"result_url": f"/results/{job_id}"})

    except Exception as e:
        now = datetime.now(timezone.utc)
        _update_job_inner(job_id, status="error",
                          error=str(e), message=f"错误: {e}", timestamp=now)
        _send_event(job_id, "error", {"error": str(e)})


# ── Jinja2 template helpers ──────────────────────────────────────
def _render(name: str, **ctx) -> HTMLResponse:
    """Simple Jinja2 template renderer."""
    from jinja2 import Environment, FileSystemLoader
    import src

    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template(name)
    ctx.setdefault("version", getattr(src, "__version__", "1.0.0"))
    return HTMLResponse(template.render(**ctx))


# ── Routes ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick health check — no template rendering."""
    return {"status": "ok", "workers": _executor._max_workers}


@app.get("/", response_class=HTMLResponse)
def upload_page():
    return _render("upload.html.jinja2", llm_enabled=_LLM_ENABLED)


@app.post("/upload")
async def upload_files(v1_file: UploadFile = File(...), v2_file: UploadFile = File(...)):
    """Accept two PDFs, create job, and start pipeline."""
    # Validate and read file contents once
    files_data: dict[str, tuple[str, bytes]] = {}
    for f, key in [(v1_file, "v1"), (v2_file, "v2")]:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            return JSONResponse({"error": f"{f.filename} 不是 PDF 文件"}, status_code=400)
        content = await f.read()
        if len(content) > 50 * 1024 * 1024:
            return JSONResponse({"error": f"{f.filename} 超过 50MB 限制"}, status_code=400)
        if len(content) < 100:
            return JSONResponse({"error": f"{f.filename} 不是有效的 PDF 文件"}, status_code=400)
        files_data[key] = (f.filename or f"{key}.pdf", content)

    job_id = uuid.uuid4().hex[:8]
    upload_dir = _PROJECT_ROOT / "data" / "uploads" / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    v1_path = upload_dir / "v1.pdf"
    v2_path = upload_dir / "v2.pdf"
    v1_path.write_bytes(files_data["v1"][1])
    v2_path.write_bytes(files_data["v2"][1])

    v1_name = files_data["v1"][0]
    v2_name = files_data["v2"][0]

    # Create job with event queue for SSE streaming
    now = datetime.now(timezone.utc)
    with _JOBS_LOCK:
        # LRU eviction if too many jobs
        if len(_JOBS) >= _MAX_JOBS:
            oldest = min(_JOBS.keys(), key=lambda k: _JOBS[k].get("created_at", datetime.min))
            del _JOBS[oldest]
        _JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "step_label": "排队中",
            "message": "任务已创建",
            "log_lines": [],
            "result_path": None,
            "error": None,
            "created_at": now,
            "v1_filename": v1_name,
            "v2_filename": v2_name,
            "v1_path": str(v1_path),
            "v2_path": str(v2_path),
            "_event_queue": queue.Queue(maxsize=500),
        }

    # Start pipeline in background
    try:
        _executor.submit(_run_pipeline, job_id, str(v1_path), str(v2_path), False)
        print(f"[UPLOAD] Job {job_id} started: {v1_name} vs {v2_name}")
    except Exception as exc:
        print(f"[UPLOAD] Failed to start pipeline: {exc}")

    return RedirectResponse(url=f"/job/{job_id}", status_code=303)


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str):
    """Processing/progress page."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        return HTMLResponse("任务未找到", status_code=404)
    return _render("job.html.jinja2", job=job)


@app.get("/api/jobs/{job_id}")
def job_api(job_id: str):
    """JSON endpoint for polling."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    elapsed = (datetime.now(timezone.utc) - job["created_at"]).total_seconds()
    return JSONResponse({
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "step_label": job["step_label"],
        "message": job.get("message", ""),
        "error": job.get("error"),
        "elapsed_seconds": int(elapsed),
    })


@app.get("/results/{job_id}", response_class=HTMLResponse)
def results_page(job_id: str, request: Request):
    """Render formatted results page."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)

    # Try to find result file (may survive server restart even if job dict doesn't)
    result_path = job.get("result_path") if job else None
    if not result_path or not os.path.exists(result_path):
        # Fallback: check data/jobs/{job_id}.json directly
        fallback = _PROJECT_ROOT / "data" / "jobs" / f"{job_id}.json"
        if fallback.exists():
            result_path = str(fallback)

    if not result_path or not os.path.exists(result_path):
        return HTMLResponse("结果文件不存在 — 可能比对尚未完成或任务已过期", status_code=404)

    if not job:
        # Reconstruct minimal job metadata from file
        job = {
            "id": job_id,
            "v1_filename": "V1",
            "v2_filename": "V2",
            "result_path": result_path,
        }

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    # Risk category name mapping
    from src.risk_classifier import RISK_CATEGORIES
    cat_map = {c["id"]: c["name"] for c in RISK_CATEGORIES}

    # Enrich changes with category names
    for change in data.get("changes", []):
        change["risk_category_names"] = [
            cat_map.get(cid, cid) for cid in change.get("risk_categories", [])
        ]
        change["_change_icon"] = {
            "added": "+", "removed": "−", "modified": "~"
        }.get(change.get("change_type", ""), "?")

    # Pre-sort frequency for template (Jinja2 reverse returns iterator, can't slice)
    freq = data.get("risk_taxonomy_snapshot", {}).get("frequency", {})
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]
    max_freq_val = sorted_freq[0][1] if sorted_freq else 1

    try:
        return _render(
            "results.html.jinja2",
            job=job,
            data=data,
            cat_map=cat_map,
            sorted_freq=sorted_freq,
            max_freq_val=max_freq_val,
            confirmed_ids=[],
        )
    except Exception as e:
        # Fallback: render a simple JSON dump page
        summary = data.get("diff_summary", {})
        changes = data.get("changes", [])
        html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>比对结果</title><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-50 p-6 font-sans"><div class="max-w-4xl mx-auto">
<h1 class="text-2xl font-bold mb-4">比对结果</h1>
<p class="text-red-600 mb-4">模板渲染失败，显示简化版: {str(e)[:200]}</p>
<div class="grid grid-cols-4 gap-3 mb-4">
<div class="bg-white rounded-xl p-4 text-center"><div class="text-2xl font-bold">{summary.get('total_changes','?')}</div><div class="text-xs text-slate-400">总差异</div></div>
<div class="bg-red-50 rounded-xl p-4 text-center"><div class="text-2xl font-bold text-red-600">{summary.get('high_risk','?')}</div><div class="text-xs text-red-500">高风险</div></div>
<div class="bg-amber-50 rounded-xl p-4 text-center"><div class="text-2xl font-bold text-amber-600">{summary.get('medium_risk','?')}</div><div class="text-xs text-amber-500">中风险</div></div>
<div class="bg-green-50 rounded-xl p-4 text-center"><div class="text-2xl font-bold text-green-600">{summary.get('low_risk','?')}</div><div class="text-xs text-green-500">低风险</div></div>
</div>"""
        for c in changes[:50]:
            html += f"""<div class="bg-white rounded-lg border border-slate-200 p-3 mb-2">
<span class="text-sm font-medium">{c.get('brief','')}</span>
<span class="text-xs text-slate-400 ml-2">[{c.get('risk_level','')}]</span>
<div class="text-xs text-red-600 mt-1">{c.get('risk_note','')[:200]}</div>
</div>"""
        html += f"<p class='text-xs text-slate-400 mt-4'>仅显示前 50 条，共 {len(changes)} 条</p></div></body></html>"
        return HTMLResponse(html, status_code=200)


@app.get("/api/results/{job_id}/json")
def results_json(job_id: str):
    """Raw JSON download."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    result_path = job.get("result_path") if job else None
    if not result_path or not os.path.exists(result_path):
        fallback = _PROJECT_ROOT / "data" / "jobs" / f"{job_id}.json"
        if fallback.exists():
            result_path = str(fallback)
    if not result_path or not os.path.exists(result_path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(json.loads(Path(result_path).read_text(encoding="utf-8")))


# ── SSE Streaming ──────────────────────────────────────────────────
@app.get("/job/{job_id}/stream")
async def job_stream(job_id: str):
    """SSE endpoint: streams changes as they're identified."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)

    eq: queue.Queue = job.get("_event_queue")
    if not eq:
        return JSONResponse({"error": "no stream"}, status_code=400)

    async def event_generator():
        import asyncio
        while True:
            try:
                msg = eq.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.3)
                yield "event: ping\ndata: {}\n\n"
                continue
            event_type = msg.get("event", "message")
            data = json.dumps(msg.get("data", {}), ensure_ascii=False)
            yield f"event: {event_type}\ndata: {data}\n\n"
            if event_type in ("done", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=False)
