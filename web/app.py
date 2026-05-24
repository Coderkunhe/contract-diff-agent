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
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

CHINA_TZ = timezone(timedelta(hours=8))

from src.config import AppConfig, get_config
from src.pipeline.extraction import extract_contract
from src.main import _run_v04

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(title="合同差异比对工具", version="1.4.0")
templates_dir = Path(__file__).resolve().parent / "templates"

# ── Config ────────────────────────────────────────────────────────
config = AppConfig.from_env()
_LLM_ENABLED = bool(config.api_key)

# ── Thread pool ───────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=config.web_workers,
                               thread_name_prefix="diffworker")


def _shutdown_executor():
    """Shutdown thread pool gracefully on exit."""
    _executor.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_executor)

# ── Job store ─────────────────────────────────────────────────────
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = config.max_jobs

_JOB_STATUS_STEPS = {
    "queued": (0, "排队中"),
    "extracting": (5, "提取 PDF 文本"),
    "tree_building": (10, "构建条款树"),
    "aligning": (20, "对齐条款"),
    "traditional_diff": (25, "传统算法对比"),
    "enhancing": (30, "LLM 增强描述"),
    "validating": (55, "L2 原文校验"),
    "classifying": (75, "交叉校验风险分类"),
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
        if "构建条款树" in line:
            stage_info = ("tree_building", 10, "构建条款树")
        elif "条款对齐" in line:
            stage_info = ("aligning", 20, "条款对齐")
        elif "传统算法对比" in line or "传统算法识别" in line:
            stage_info = ("traditional_diff", 25, "传统算法对比")
        elif "LLM 增强描述" in line or "LLM增强" in line:
            stage_info = ("enhancing", 30, "LLM 增强描述")
        elif "校验" in line or "原文校验" in line:
            stage_info = ("validating", 55, "L2 原文校验")
        elif "风险分类" in line or "交叉校验" in line:
            stage_info = ("classifying", 75, "交叉校验风险分类")
        elif "离线模式" in line:
            stage_info = ("validating", 25, "离线模式跳过 LLM")

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
            "human_note": change.get("human_note"),
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

        offline = not bool(config.api_key)
        thorough = job.get("thorough", False) if job else False
        fake_args = argparse.Namespace(validate=thorough, thorough=thorough, offline=offline)

        _send_event(job_id, "progress", {"status": "tree_building", "step": "构建条款树",
                      "progress": 12, "message": "正在解析合同条款结构..."})

        # Capture stdout for progress tracking
        tracker = JobProgressIO(job_id, sys.stdout)
        old_stdout = sys.stdout
        sys.stdout = tracker

        try:
            result = _run_v04(
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

        # Self-evolution: extract and save learning (non-fatal)
        try:
            from src.pipeline.learning import extract_learning, save_learning
            learning = extract_learning(result, job_id)
            save_learning(learning)
        except Exception as e:
            print(f"[LEARNING] Failed to save learning for {job_id}: {e}")

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
    return HTMLResponse(
        template.render(**ctx),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ── Routes ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick health check — no template rendering."""
    return {"status": "ok", "workers": _executor._max_workers}


@app.get("/", response_class=HTMLResponse)
def upload_page():
    return _render("upload.html.jinja2", llm_enabled=_LLM_ENABLED)


@app.post("/upload")
async def upload_files(v1_file: UploadFile = File(...), v2_file: UploadFile = File(...),
                       thorough: bool = Form(False)):
    """Accept two PDFs, create job, and start pipeline."""
    # Validate and read file contents once
    files_data: dict[str, tuple[str, bytes]] = {}
    for f, key in [(v1_file, "v1"), (v2_file, "v2")]:
        if not f.filename or not f.filename.lower().endswith((".pdf", ".docx")):
            return JSONResponse({"error": f"{f.filename} 不是 PDF/DOCX 文件"}, status_code=400)
        content = await f.read()
        if len(content) > 50 * 1024 * 1024:
            return JSONResponse({"error": f"{f.filename} 超过 50MB 限制"}, status_code=400)
        if len(content) < 100:
            return JSONResponse({"error": f"{f.filename} 不是有效的 PDF 文件"}, status_code=400)
        files_data[key] = (f.filename or f"{key}.pdf", content)

    job_id = uuid.uuid4().hex[:8]
    upload_dir = _PROJECT_ROOT / "data" / "uploads" / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Preserve original file extension for format detection
    v1_name = files_data["v1"][0]
    v2_name = files_data["v2"][0]
    v1_ext = os.path.splitext(v1_name)[1].lower() or ".pdf"
    v2_ext = os.path.splitext(v2_name)[1].lower() or ".pdf"
    v1_path = upload_dir / f"v1{v1_ext}"
    v2_path = upload_dir / f"v2{v2_ext}"
    v1_path.write_bytes(files_data["v1"][1])
    v2_path.write_bytes(files_data["v2"][1])

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
            "thorough": thorough,
        }

    # Start pipeline in background
    try:
        _executor.submit(_run_pipeline, job_id, str(v1_path), str(v2_path), False)
        print(f"[UPLOAD] Job {job_id} started: {v1_name} vs {v2_name}")
    except Exception as exc:
        print(f"[UPLOAD] Failed to start pipeline: {exc}")

    return RedirectResponse(url=f"/job/{job_id}", status_code=303)


@app.get("/demo")
def demo_start():
    """One-click demo: auto-start comparison with built-in sample contracts."""
    sample_v1 = _PROJECT_ROOT / "docs" / "天猫服务协议2015(2).pdf"
    sample_v2 = _PROJECT_ROOT / "docs" / "天猫服务协议2026(2).pdf"

    if not sample_v1.exists() or not sample_v2.exists():
        return HTMLResponse("Sample PDFs not found. Run from repo root.", status_code=500)

    job_id = uuid.uuid4().hex[:8]
    upload_dir = _PROJECT_ROOT / "data" / "uploads" / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(sample_v1, upload_dir / "v1.pdf")
    shutil.copy2(sample_v2, upload_dir / "v2.pdf")

    now = datetime.now(timezone.utc)
    with _JOBS_LOCK:
        if len(_JOBS) >= _MAX_JOBS:
            oldest = min(_JOBS.keys(), key=lambda k: _JOBS[k].get("created_at", datetime.min))
            del _JOBS[oldest]
        _JOBS[job_id] = {
            "id": job_id, "status": "queued", "progress": 0,
            "step_label": "排队中", "message": "快速体验已启动",
            "log_lines": [], "result_path": None, "error": None,
            "created_at": now,
            "v1_filename": "天猫服务协议2015 (V1)",
            "v2_filename": "天猫服务协议2026 (V2)",
            "v1_path": str(upload_dir / "v1.pdf"),
            "v2_path": str(upload_dir / "v2.pdf"),
            "_event_queue": queue.Queue(maxsize=500),
            "thorough": False,
        }

    try:
        _executor.submit(_run_pipeline, job_id,
                        str(upload_dir / "v1.pdf"),
                        str(upload_dir / "v2.pdf"),
                        False)
        print(f"[DEMO] Job {job_id} started: one-click demo")
    except Exception as exc:
        print(f"[DEMO] Failed to start pipeline: {exc}")

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
    from src.constants.risks import RISK_CATEGORIES
    cat_map = {c["id"]: c["name"] for c in RISK_CATEGORIES}

    # Enrich changes with category names, ensure risk_categories exists
    for change in data.get("changes", []):
        if not change.get("risk_categories"):
            change["risk_categories"] = []
        change["risk_category_names"] = [
            cat_map.get(cid, cid) for cid in change["risk_categories"]
        ]
        change["_change_icon"] = {
            "added": "+", "removed": "−", "modified": "~"
        }.get(change.get("change_type", ""), "?")

    # Pre-sort frequency for template (Jinja2 reverse returns iterator, can't slice)
    freq = data.get("risk_taxonomy_snapshot", {}).get("frequency", {})
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]
    max_freq_val = sorted_freq[0][1] if sorted_freq else 1

    # Ensure risk taxonomy snapshot has categories for template rendering
    rts = data.get("risk_taxonomy_snapshot", {})
    if not rts.get("categories"):
        rts["categories"] = RISK_CATEGORIES

    # Extract existing human verdicts for UI init
    verdicts = {}
    for change in data.get("changes", []):
        hv = change.get("human_verdict")
        if hv:
            verdicts[change.get("id", "")] = {
                "action": hv.get("action", "confirmed"),
                "corrected_risk_level": hv.get("corrected_risk_level"),
                "corrected_risk_categories": hv.get("corrected_risk_categories"),
                "corrected_risk_note": hv.get("corrected_risk_note"),
            }

    try:
        return _render(
            "results.html.jinja2",
            job=job,
            data=data,
            cat_map=cat_map,
            sorted_freq=sorted_freq,
            max_freq_val=max_freq_val,
            verdicts_json=json.dumps(verdicts, ensure_ascii=False),
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


@app.get("/api/results/{job_id}/pdf")
def results_pdf(job_id: str, ids: str = ""):
    try:
        return _generate_pdf(job_id, ids)
    except Exception as e:
        return JSONResponse({"error": f"PDF 生成失败: {str(e)[:200]}"}, status_code=500)


def _find_cjk_font() -> str | None:
    """Find an available CJK font across platforms (macOS, Linux, Docker)."""
    import platform
    paths: list[str] = []

    system = platform.system()
    if system == "Darwin":
        paths = [
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:  # Linux (including Docker)
        paths = [
            # Noto Sans CJK (most common in Docker images)
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
            # WQY ZenHei (common fallback)
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            # Droid Sans Fallback
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            # AR PL UMing
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]

    # Also try any font in the list regardless of platform
    common = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for p in common:
        if p not in paths:
            paths.append(p)

    for fp in paths:
        if os.path.exists(fp):
            return fp
    return None


def _map_v2_pages(v2_path: str, changes: list[dict]) -> dict[str, int]:
    """Map change IDs to V2 page numbers by matching v2_snippet against V2 PDF.

    Used as fallback when pipeline didn't already compute v2_page.
    """
    if not v2_path or not os.path.exists(v2_path):
        return {}

    try:
        v2_doc = extract_contract(v2_path, keep_english=True)
        page_texts = [p.text for p in v2_doc.pages]
    except Exception:
        try:
            import pdfplumber
            page_texts = []
            with pdfplumber.open(v2_path) as pdf:
                for page in pdf.pages:
                    page_texts.append(page.extract_text() or "")
        except Exception:
            return {}

    if not page_texts:
        return {}

    def _norm(t):
        return "".join(t.split())

    mapping: dict[str, int] = {}
    for c in changes:
        snippet = (c.get("v2_snippet") or "").strip()
        if not snippet:
            continue
        norm_s = _norm(snippet)
        cid = c.get("id", "")
        for pi, pt in enumerate(page_texts):
            if snippet in pt:
                mapping[cid] = pi + 1
                break
            if norm_s and norm_s in _norm(pt):
                mapping[cid] = pi + 1
                break
            if len(norm_s) >= 40 and norm_s[:40] in _norm(pt):
                mapping[cid] = pi + 1
                break

    return mapping


def _render_page_paragraphs(pdf, page_text: str):
    """Render a page's text as properly formatted paragraphs.

    Universal merge-then-render: mid-sentence line breaks (lines not ending
    with sentence-ending punctuation) are always merged, regardless of source
    format.  Blank lines serve as paragraph separators when present; when
    absent the whole page is one logical paragraph.
    """
    import re

    raw_lines = page_text.split("\n")
    sentence_ends = {"。", "；", "）", ")", "》", "？", "！", "：", ":", "”"}

    # Regex for section headers
    section_pattern = re.compile(
        r"^[（(]?[一二三四五六七八九十]+[）)]?\s*[、．.]\s*"
        r"|^第[一二三四五六七八九十\d]+[章节条]\s*"
    )

    # ①  Group into paragraphs (blank-line separation when present)
    has_blank_lines = any(not l.strip() for l in raw_lines)
    paragraphs: list[list[str]] = []
    if has_blank_lines:
        current: list[str] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    paragraphs.append(current)
                    current = []
            else:
                current.append(stripped)
        if current:
            paragraphs.append(current)
    else:
        # No blank lines — treat the whole page as one paragraph
        non_empty = [l.strip() for l in raw_lines if l.strip()]
        if non_empty:
            paragraphs = [non_empty]

    # ②  Merge mid-sentence breaks within each paragraph, then render
    cell_w = pdf.epw
    for pi, para_lines in enumerate(paragraphs):
        if pi > 0:
            pdf.ln(3)

        merged: list[str] = []
        for line in para_lines:
            if merged and merged[-1] and merged[-1][-1] not in sentence_ends:
                # Join with space only between two Latin-script boundaries
                # to avoid "thisAgreement" while keeping Chinese "为一体".
                prev_ch = merged[-1][-1]
                curr_ch = line[0] if line else ""
                sep = " " if (prev_ch.isascii() and prev_ch.isalpha() and
                              curr_ch.isascii() and curr_ch.isalpha()) else ""
                merged[-1] += sep + line
            else:
                merged.append(line)
        full_text = "".join(merged)

        is_header = bool(section_pattern.match(full_text)) and len(full_text) < 80
        if is_header:
            pdf.set_font("CJK", "", 11)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(cell_w, 7, full_text)
            pdf.ln(1)
        else:
            pdf.set_font("CJK", "", 9)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(cell_w, 5.5, full_text)

    pdf.set_text_color(0, 0, 0)


def _render_v2_original(
    pdf,
    v2_path: str | None,
    v2_name: str,
    pdf_pages: dict[int, int],
):
    """Render V2 original contract text page by page.

    Tracks which PDF page each original V2 page lands on so diff items
    can link back to the correct page. Supports both PDF and DOCX.
    """
    if not v2_path or not os.path.exists(v2_path):
        return

    # Extract V2 page texts using the same pipeline extraction (handles PDF + DOCX)
    try:
        v2_doc = extract_contract(v2_path, keep_english=True)
        v2_page_texts = [p.text for p in v2_doc.pages]
    except Exception:
        # Fallback: try pdfplumber for PDFs
        try:
            import pdfplumber
            v2_page_texts = []
            with pdfplumber.open(v2_path) as v2_pdf:
                for page in v2_pdf.pages:
                    v2_page_texts.append(page.extract_text() or "")
        except Exception:
            # Last resort: treat as single-page plain text
            try:
                text = Path(v2_path).read_text(encoding="utf-8", errors="replace")
                v2_page_texts = [text]
            except Exception:
                v2_page_texts = []

    if not v2_page_texts:
        # Render fallback message
        pdf.ln(10)
        pdf.set_font("CJK", "", 14)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 10, "（无法提取原文内容）", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        return

    # ── Section header ──
    pdf.ln(4)
    pdf.set_font("CJK", "", 20)
    pdf.cell(0, 12, "V2 合同原文", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("CJK", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, f"文件: {v2_name}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"共 {len(v2_page_texts)} 页", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)
    pdf.set_draw_color(52, 73, 245)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin + 40, pdf.get_y(), pdf.w - pdf.r_margin - 40, pdf.get_y())
    pdf.ln(8)

    # ── Render each V2 page ──
    for i, page_text in enumerate(v2_page_texts):
        orig_page = i + 1

        # Ensure page label lands at top of a page when space is tight.
        # If there's < 80 mm left, start a fresh page (a typical V2 page
        # needs ~14 lines × 5.5mm ≈ 77mm).
        space_avail = pdf.h - pdf.b_margin - pdf.get_y()
        if space_avail < 80:
            pdf.add_page()

        # Record current PDF page for link targeting
        pdf_pages[orig_page] = pdf.page

        # Page label — prominent header with rules above & below
        if i > 0:
            pdf.ln(2)
        pdf.set_draw_color(200, 210, 225)
        pdf.set_line_width(0.5)
        pdf.line(pdf.l_margin + 60, pdf.get_y(), pdf.w - pdf.r_margin - 60, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("CJK", "", 11)
        pdf.set_text_color(80, 90, 110)
        pdf.cell(0, 8, f"V2 第 {orig_page} 页", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        pdf.set_draw_color(200, 210, 225)
        pdf.set_line_width(0.3)
        pdf.line(pdf.l_margin + 60, pdf.get_y(), pdf.w - pdf.r_margin - 60, pdf.get_y())
        pdf.ln(5)

        # Render paragraphs with proper formatting
        _render_page_paragraphs(pdf, page_text)
        pdf.ln(4)


def _generate_pdf(job_id: str, ids: str):
    """Generate a PDF report of (confirmed) changes."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    result_path = job.get("result_path") if job else None
    if not result_path or not os.path.exists(result_path):
        fallback = _PROJECT_ROOT / "data" / "jobs" / f"{job_id}.json"
        if fallback.exists():
            result_path = str(fallback)
    if not result_path or not os.path.exists(result_path):
        return JSONResponse({"error": "not found"}, status_code=404)

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    changes = data.get("changes", [])
    # Normalize: ensure all changes have an id field
    for i, c in enumerate(changes):
        if "id" not in c:
            c["id"] = f"diff-{i+1:03d}"

    confirmed_ids = [x for x in ids.split(",") if x] if ids else [c["id"] for c in changes]
    filtered = [c for c in changes if c.get("id") in confirmed_ids]

    # ── V2 page mapping ──
    # Prefer pipeline-computed v2_page; fall back to runtime matching
    v2_path = job.get("v2_path") if job else None
    # Fallback: read v2_path from result JSON meta (job dict may not survive restart)
    if not v2_path:
        v2_path = data.get("meta", {}).get("contract_v2")
    if v2_path and not os.path.exists(v2_path):
        v2_path = None
    v2_page_map: dict[str, int] = {}
    for c in filtered:
        if c.get("v2_page"):
            v2_page_map[c["id"]] = c["v2_page"]
    missing = [c for c in filtered if c["id"] not in v2_page_map]
    if missing and v2_path:
        fallback_map = _map_v2_pages(v2_path, missing)
        v2_page_map.update(fallback_map)
        for c in missing:
            if c["id"] in fallback_map:
                c["v2_page"] = fallback_map[c["id"]]


    from fpdf import FPDF

    font_path = _find_cjk_font()
    if not font_path:
        return JSONResponse(
            {"error": "PDF 生成失败: 未找到中文字体。请安装 Noto Sans CJK 或 WQY ZenHei"},
            status_code=500,
        )

    pdf = FPDF()
    pdf.add_font("CJK", "", font_path)

    # Build filename from contract names
    v1_name = job.get("v1_filename", "V1") if job else "V1"
    v2_name = job.get("v2_filename", "V2") if job else "V2"
    v1_base = v1_name.rsplit(".", 1)[0] if "." in v1_name else v1_name
    v2_base = v2_name.rsplit(".", 1)[0] if "." in v2_name else v2_name
    safe_v1 = v1_base.replace(" ", "_")[:30]
    safe_v2 = v2_base.replace(" ", "_")[:30]
    pdf_filename = f"contract_diff_report.pdf"  # ASCII-safe filename
    confirmed_at = datetime.now(CHINA_TZ).strftime("%Y-%m-%d %H:%M 北京时间")

    pdf.add_page()
    pdf.set_auto_page_break(True, 20)

    # ── ① Cover page ──
    pdf.ln(20)
    pdf.set_font("CJK", "", 24)
    pdf.cell(0, 14, "合同差异比对报告", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("CJK", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"确认时间: {confirmed_at}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.cell(0, 8, f"对比文件: {v1_name}  vs  {v2_name}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_draw_color(52, 73, 245)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin + 40, pdf.get_y(), pdf.w - pdf.r_margin - 40, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("CJK", "", 10)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 7, "本报告由合同慧眼自动生成", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    # ── ② TOC page ──
    toc_v2_link = pdf.add_link()
    toc_diff_link = pdf.add_link()

    pdf.add_page()
    pdf.ln(16)
    pdf.set_font("CJK", "", 24)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 16, "目  录", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)
    # Decorative rule
    pdf.set_draw_color(52, 73, 245)
    pdf.set_line_width(1.0)
    y_rule = pdf.get_y()
    pdf.line(pdf.l_margin + 50, y_rule, pdf.w - pdf.r_margin - 50, y_rule)
    pdf.ln(14)

    # Entry 1: V2 original
    pdf.set_font("CJK", "", 16)
    pdf.set_text_color(52, 73, 245)
    pdf.cell(0, 12, "1    V2 合同原文", link=toc_v2_link, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("CJK", "", 10)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 7, f"     {v2_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.ln(16)
    pdf.set_draw_color(220, 220, 220)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin + 30, pdf.get_y(), pdf.w - pdf.r_margin - 30, pdf.get_y())
    pdf.ln(14)

    # Entry 2: Diff report
    pdf.set_font("CJK", "", 16)
    pdf.set_text_color(52, 73, 245)
    pdf.cell(0, 12, "2    合同差异比对报告", link=toc_diff_link, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("CJK", "", 10)
    pdf.set_text_color(140, 140, 140)
    s = data.get("diff_summary", {})
    pdf.cell(0, 7, f"     {s.get('total_changes', len(changes))} 条差异，{s.get('high_risk', 0)} 条高风险", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.ln(20)
    pdf.set_font("CJK", "", 9)
    pdf.set_text_color(180, 180, 180)
    pdf.cell(0, 6, "点击上方条目可直接跳转至对应章节", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    # ── ③ V2 original text ──
    v2_pdf_pages: dict[int, int] = {}  # v2 original page → pdf page
    pdf.add_page()
    pdf.set_link(toc_v2_link, page=pdf.page)
    _render_v2_original(pdf, v2_path, v2_name, v2_pdf_pages)

    # ── ④ Diff report ──
    pdf.add_page()
    pdf.set_link(toc_diff_link, page=pdf.page)

    pdf.ln(4)
    pdf.set_font("CJK", "", 20)
    pdf.cell(0, 12, "合同差异比对报告", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("CJK", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, f"确认时间: {confirmed_at}  |  {v1_name}  vs  {v2_name}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)
    pdf.set_draw_color(52, 73, 245)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin + 40, pdf.get_y(), pdf.w - pdf.r_margin - 40, pdf.get_y())
    pdf.ln(6)

    # ── Summary ──
    pdf.set_font("CJK", "", 13)
    pdf.cell(0, 9, "比对概要", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("CJK", "", 11)
    pdf.cell(0, 8, f"总差异: {s.get('total_changes', len(changes))} 条  |  "
                   f"高风险: {s.get('high_risk', '?')}  |  "
                   f"中风险: {s.get('medium_risk', '?')}  |  "
                   f"低风险: {s.get('low_risk', '?')}  |  "
                   f"已确认: {len(filtered)} 条",
           new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ── Each change ──
    seq = 0
    for ci, c in enumerate(filtered):
        seq += 1
        risk = c.get("risk_level", "low")
        risk_label = {"high": "高风险", "medium": "中风险", "low": "低风险"}.get(risk, "")
        type_label = {"added": "新增", "removed": "删除", "modified": "修改"}.get(c.get("change_type", ""), "")

        # Light separator between items
        if ci > 0:
            pdf.ln(4)
            pdf.set_draw_color(220, 220, 220)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(5)

        # Item header
        pdf.set_font("CJK", "", 13)
        if risk == "high":
            pdf.set_text_color(180, 40, 40)
        elif risk == "medium":
            pdf.set_text_color(180, 120, 20)
        else:
            pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 8, f"{seq}. [{type_label}] [{risk_label}]", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        # Clause ref + V2 page link
        ref_text = c.get("clause_ref_v2") or c.get("clause_ref_v1", "")
        v2_page = v2_page_map.get(c.get("id", ""))
        if ref_text or v2_page:
            pdf.set_font("CJK", "", 9)
            pdf.set_text_color(120, 120, 120)
            line_text = f"> {ref_text}" if ref_text else ""
            pdf.cell(0, 6, line_text, new_x="LMARGIN", new_y="NEXT")
            if v2_page:
                link_id = pdf.add_link()
                target_pdf_page = v2_pdf_pages.get(v2_page)
                if target_pdf_page:
                    pdf.set_link(link_id, page=target_pdf_page)
                pdf.set_font("CJK", "", 9)
                pdf.set_text_color(52, 73, 245)
                pdf.cell(0, 6, f"  → V2 第{v2_page}页", link=link_id, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        # Brief
        pdf.set_font("CJK", "", 11)
        pdf.multi_cell(0, 7, c.get("brief", "")[:300])
        pdf.ln(2)

        # Snippets
        vs1 = c.get("v1_snippet", "")
        vs2 = c.get("v2_snippet", "")
        if vs1 or vs2:
            pdf.set_font("CJK", "", 9)
            pdf.set_fill_color(248, 248, 250)
            if vs1:
                pdf.cell(0, 6, f"  V1: {vs1[:250]}", new_x="LMARGIN", new_y="NEXT", fill=True)
            if vs2:
                pdf.cell(0, 6, f"  V2: {vs2[:250]}", new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.ln(2)

        # Risk note
        note = c.get("risk_note", "")
        if note:
            pdf.set_font("CJK", "", 10)
            if risk == "high":
                pdf.set_text_color(180, 40, 40)
            elif risk == "medium":
                pdf.set_text_color(180, 120, 20)
            else:
                pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 6, f"[!] 风险提示: {note}"[:300])
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

        # Human note
        hn = c.get("human_note", "")
        if hn:
            pdf.set_font("CJK", "", 9)
            pdf.set_text_color(80, 80, 160)
            pdf.multi_cell(0, 6, f"[备注] {hn}"[:300])
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

        # Confirmation time
        pdf.set_font("CJK", "", 8)
        pdf.set_text_color(160, 160, 160)
        pdf.cell(0, 5, f"确认于 {confirmed_at}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    pdf_bytes = bytes(pdf.output())
    from urllib.parse import quote
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition":
                             f"attachment; filename=\"{pdf_filename}\"; "
                             f"filename*=UTF-8''{quote(pdf_filename)}"})


@app.put("/api/results/{job_id}/notes")
async def update_notes(job_id: str, request: Request):
    """Save human notes for confirmed changes."""
    body = await request.json()
    notes = body.get("notes", {})  # {change_id: note_text}

    result_path = None
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job:
            result_path = job.get("result_path")
    if not result_path:
        fallback = _PROJECT_ROOT / "data" / "jobs" / f"{job_id}.json"
        if fallback.exists():
            result_path = str(fallback)
    if not result_path or not os.path.exists(result_path):
        return JSONResponse({"error": "not found"}, status_code=404)

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    for change in data.get("changes", []):
        cid = change.get("id", "")
        if cid in notes:
            change["human_note"] = notes[cid]
            updated += 1

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return JSONResponse({"updated": updated})


@app.put("/api/results/{job_id}/verdicts")
async def update_verdicts(job_id: str, request: Request):
    """Save human verdicts (confirmed/rejected/corrected) for changes.

    Accepts: {"verdicts": {change_id: {action, corrected_risk_level?,
               corrected_risk_categories?, corrected_risk_note?}}}

    Snapshots original LLM output before overwriting risk fields.
    """
    body = await request.json()
    verdicts = body.get("verdicts", {})

    result_path = None
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job:
            result_path = job.get("result_path")
    if not result_path:
        fallback = _PROJECT_ROOT / "data" / "jobs" / f"{job_id}.json"
        if fallback.exists():
            result_path = str(fallback)
    if not result_path or not os.path.exists(result_path):
        return JSONResponse({"error": "not found"}, status_code=404)

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    corrections = 0
    now_ts = datetime.now(CHINA_TZ).isoformat()

    for change in data.get("changes", []):
        cid = change.get("id", "")
        if cid not in verdicts:
            continue
        v = verdicts[cid]
        action = v.get("action", "confirmed")

        # Snapshot original LLM output if first time
        existing = change.get("human_verdict") or {}
        if not existing:
            existing["original"] = {
                "risk_level": change.get("risk_level", "medium"),
                "risk_categories": list(change.get("risk_categories", []) or []),
                "risk_note": change.get("risk_note", ""),
            }

        existing["action"] = action
        existing["timestamp"] = existing.get("timestamp") or now_ts

        if action == "corrected":
            corrections += 1
            if "corrected_risk_level" in v:
                change["risk_level"] = v["corrected_risk_level"]
                existing["corrected_risk_level"] = v["corrected_risk_level"]
            if "corrected_risk_categories" in v:
                change["risk_categories"] = v["corrected_risk_categories"]
                existing["corrected_risk_categories"] = v["corrected_risk_categories"]
            if "corrected_risk_note" in v:
                change["risk_note"] = v["corrected_risk_note"]
                existing["corrected_risk_note"] = v["corrected_risk_note"]

        change["human_verdict"] = existing
        updated += 1

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Re-extract learnings so verdict feedback shows in /learnings
    try:
        from src.pipeline.learning import extract_learning, save_learning
        learning = extract_learning(data, job_id)
        save_learning(learning)
    except Exception as e:
        print(f"[LEARNING] 判决后重新提取学习数据失败: {e}")

    return JSONResponse({"updated": updated, "corrections": corrections})


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


@app.get("/learnings", response_class=HTMLResponse)
def learnings_page():
    """Self-evolution history page."""
    from src.pipeline.learning import _load_index, _resolve_dir
    import json as _json
    data_dir = _resolve_dir(None)
    index = _load_index(data_dir)

    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("learnings.html.jinja2")
    html = template.render(
        runs_json=_json.dumps(index.get("runs", []), ensure_ascii=False),
        trends_json=_json.dumps(index.get("global_trends", {}), ensure_ascii=False),
        total_runs=index.get("total_runs", 0),
        updated_at=index.get("updated_at", ""),
        version=getattr(__import__("src"), "__version__", "1.0.0"),
    )
    return HTMLResponse(html, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/api/learnings", response_class=JSONResponse)
def api_learnings():
    """Raw learning index JSON — bypasses template for foolproof verification."""
    from src.pipeline.learning import _load_index, _resolve_dir
    index = _load_index(_resolve_dir(None))
    return JSONResponse(index, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=False)
