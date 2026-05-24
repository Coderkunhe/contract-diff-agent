"""Self-evolution learning extractor.

After each pipeline run, extracts learnings from the result JSON
and persists them to data/learnings/. Zero new dependencies.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.constants.risks import RISK_CATEGORIES

_CAT_NAMES: dict[str, str] = {c["id"]: c["name"] for c in RISK_CATEGORIES}
_CAT_REVERSE: dict[str, str] = {c["name"]: c["id"] for c in RISK_CATEGORIES}
_LEARNINGS_DIR = Path("data/learnings")
_INDEX_LOCK = threading.Lock()


def extract_learning(result: dict, job_id: str) -> dict:
    """Extract a learning record from a pipeline result dict."""
    meta = result.get("meta", {})
    summary = result.get("diff_summary", {})
    changes = result.get("changes", [])
    taxonomy = result.get("risk_taxonomy_snapshot", {})

    total = summary.get("total_changes", len(changes))
    high = summary.get("high_risk", 0)
    medium = summary.get("medium_risk", 0)
    low = summary.get("low_risk", 0)

    # Category distribution from taxonomy, reverse-map names to IDs
    freq = taxonomy.get("frequency", {})
    cat_dist: dict[str, int] = {}
    for name, count in freq.items():
        cid = _CAT_REVERSE.get(name, name)
        cat_dist[cid] = count
    top_categories = sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)[:5]
    top_categories_out = [
        {"id": cid, "name": _CAT_NAMES.get(cid, cid), "count": count}
        for cid, count in top_categories
    ]

    # Quality signals from validation sub-objects
    rejected = 0
    uncertain = 0
    high_conf_verified = 0
    human_count = 0
    human_notes: list[dict] = []

    # High-confidence pattern grouping
    pattern_groups: dict[tuple[str, str], list[str]] = {}

    for c in changes:
        v = c.get("validation", {}) or {}
        l3 = v.get("l3_verdict", "")
        conf = v.get("confidence", 0) or 0
        status = v.get("status", "")

        if l3 == "rejected":
            rejected += 1
        elif l3 == "uncertain":
            uncertain += 1

        if conf > 0.9 and status == "verified":
            high_conf_verified += 1
            for cat in c.get("risk_categories", []) or []:
                rl = c.get("risk_level", "low")
                key = (cat, rl)
                if key not in pattern_groups:
                    pattern_groups[key] = []
                brief = c.get("brief", "")
                if brief and len(pattern_groups[key]) < 3:
                    pattern_groups[key].append(brief)

        note = c.get("human_note", "")
        if note:
            human_count += 1
            human_notes.append({
                "change_id": c.get("id", ""),
                "note_preview": note[:80],
                "risk_level": c.get("risk_level", "low"),
            })

    # Build high-confidence patterns (top 10 by count)
    hc_patterns = []
    for (cat, rl), briefs in sorted(pattern_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        hc_patterns.append({
            "category": cat,
            "risk_level": rl,
            "pattern": briefs[0][:60] if briefs else "",
            "count": len(briefs),
        })

    validated = rejected + uncertain + high_conf_verified
    rejection_rate = rejected / max(validated, 1)
    uncertain_rate = uncertain / max(validated, 1)

    # Build summary string
    model_str = meta.get("model") or "offline"
    pipeline = meta.get("pipeline", "")
    parts = [f"{pipeline}({model_str}): {total}条差异"]
    if high:
        parts.append(f"{high}条高风险")
    if top_categories_out:
        tc = top_categories_out[0]
        parts.append(f"{tc['name']}{tc['count']}条")
    summary_str = ", ".join(parts)

    return {
        "job_id": job_id,
        "timestamp": meta.get("compared_at") or datetime.now(timezone.utc).isoformat(),
        "meta": {
            "agent_version": meta.get("agent_version", ""),
            "pipeline": pipeline,
            "model": model_str,
            "token_estimate": meta.get("token_estimate", {}),
        },
        "risk_profile": {
            "total_changes": total,
            "high_risk_count": high,
            "medium_risk_count": medium,
            "low_risk_count": low,
            "top_categories": top_categories_out,
            "category_distribution": cat_dist,
        },
        "quality_signals": {
            "validation_rejection_rate": round(rejection_rate, 4),
            "validation_uncertain_rate": round(uncertain_rate, 4),
            "human_corrections_count": human_count,
            "high_confidence_verified_count": high_conf_verified,
        },
        "high_confidence_patterns": hc_patterns,
        "human_notes": human_notes[:20],
        "summary": summary_str,
    }


# ── Persistence ────────────────────────────────────────────────────

def _resolve_dir(data_dir: Optional[Path] = None) -> Path:
    """Resolve data_dir, making relative paths project-root-relative."""
    d = data_dir or _LEARNINGS_DIR
    if not d.is_absolute():
        # Find project root (2 levels up from src/pipeline/)
        project_root = Path(__file__).resolve().parent.parent.parent
        d = project_root / d
    return d


def _load_index(data_dir: Path) -> dict:
    idx_path = data_dir / "index.json"
    if not idx_path.exists():
        return {"version": "1.0", "total_runs": 0, "runs": [], "global_trends": {}}
    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"version": "1.0", "total_runs": 0, "runs": [], "global_trends": {}}


def _save_index(index: dict, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    idx_path = data_dir / "index.json"
    tmp = idx_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    os.replace(tmp, idx_path)


def _update_index(learning: dict, data_dir: Path) -> dict:
    index = _load_index(data_dir)
    runs = index.get("runs", [])

    # Upsert by job_id
    existing = {r["job_id"]: i for i, r in enumerate(runs)}
    run_entry = {
        "job_id": learning["job_id"],
        "timestamp": learning["timestamp"],
        "pipeline": learning["meta"]["pipeline"],
        "model": learning["meta"]["model"],
        "total_changes": learning["risk_profile"]["total_changes"],
        "high_risk": learning["risk_profile"]["high_risk_count"],
        "medium_risk": learning["risk_profile"]["medium_risk_count"],
        "low_risk": learning["risk_profile"]["low_risk_count"],
        "top_category": learning["risk_profile"]["top_categories"][0] if learning["risk_profile"]["top_categories"] else None,
        "top_categories": learning["risk_profile"]["top_categories"],
        "validation_rejection_rate": learning["quality_signals"]["validation_rejection_rate"],
        "human_corrections_count": learning["quality_signals"]["human_corrections_count"],
        "summary": learning["summary"],
    }

    if learning["job_id"] in existing:
        runs[existing[learning["job_id"]]] = run_entry
    else:
        runs.append(run_entry)

    runs.sort(key=lambda r: r["timestamp"], reverse=True)
    index["runs"] = runs
    index["total_runs"] = len(runs)
    index["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Recompute global trends
    if runs:
        cat_totals: dict[str, float] = {}
        total_changes_sum = 0
        high_sum = 0
        rej_sum = 0.0
        rej_count = 0
        human_total = 0
        for r in runs:
            total_changes_sum += r["total_changes"]
            high_sum += r["high_risk"]
            human_total += r["human_corrections_count"]
            if r["validation_rejection_rate"] is not None:
                rej_sum += r["validation_rejection_rate"]
                rej_count += 1
            for tc in r.get("top_categories", []) or []:
                name = tc["name"]
                cat_totals[name] = cat_totals.get(name, 0) + tc["count"]

        top_cats = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        n = len(runs)
        index["global_trends"] = {
            "top_risk_categories": [
                {"name": name, "total_count": total} for name, total in top_cats
            ],
            "avg_high_risk_count": round(high_sum / n, 1),
            "avg_total_changes": round(total_changes_sum / n, 1),
            "avg_rejection_rate": round(rej_sum / max(rej_count, 1), 4),
            "total_human_corrections": human_total,
        }

    return index


def save_learning(learning: dict, data_dir: Optional[Path] = None) -> Path:
    """Persist a learning record and update the index. Thread-safe via lock."""
    d = _resolve_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)

    run_path = d / f"run-{learning['job_id']}.json"
    tmp = run_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(learning, f, ensure_ascii=False, indent=2)
    os.replace(tmp, run_path)

    with _INDEX_LOCK:
        index = _update_index(learning, d)
        _save_index(index, d)

    return run_path


# ── Prompt injection helpers ───────────────────────────────────────

def load_past_learnings(limit: int = 5, data_dir: Optional[Path] = None) -> list[dict]:
    """Return the most recent run summaries from the index (top N)."""
    d = _resolve_dir(data_dir)
    index = _load_index(d)
    return index.get("runs", [])[:limit]


def format_learnings_context(learnings: list[dict]) -> str:
    """Format past learnings as a compact prompt context block."""
    if not learnings:
        return ""

    lines = [f"## 历史比对经验（基于 {len(learnings)} 次过往合同比对）", ""]
    lines.append("过往比对中常见的风险模式：")

    for i, l in enumerate(learnings, 1):
        ts = l.get("timestamp", "")[:16].replace("T", " ")
        model = l.get("model") or "offline"
        tc = l.get("top_category")
        top_cat = f"{tc['name']}({tc['count']}条)" if tc else ""
        lines.append(
            f"- 第{i}次({ts}): {top_cat} "
            f"高风险{l['high_risk']}条, 中风险{l['medium_risk']}条, "
            f"LLM校验拒绝率{_rate_pct(l.get('validation_rejection_rate'))}"
        )

    # Highlight top recurring categories across all runs
    all_cats: dict[str, int] = {}
    for l in learnings:
        for tc in l.get("top_categories", []) or []:
            all_cats[tc["name"]] = all_cats.get(tc["name"], 0) + tc["count"]
    top = sorted(all_cats.items(), key=lambda x: x[1], reverse=True)[:3]
    if top:
        lines.append("")
        lines.append(
            "重点关注: " + "、".join(f"{name}({total}次)" for name, total in top)
            + " 在过往比对中频繁出现。"
        )
        lines.append("在分析当前合同差异时，请优先检查这些高风险类别。")

    return "\n".join(lines)


def _rate_pct(rate) -> str:
    if rate is None or rate == 0:
        return "N/A"
    return f"{rate * 100:.0f}%"
