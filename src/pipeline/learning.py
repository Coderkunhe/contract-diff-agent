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
    """Extract a learning record from a pipeline result dict.

    Reads LLM auto-validation quality signals, high-confidence patterns,
    human notes, AND human verdicts (confirmed/rejected/corrected).
    """
    meta = result.get("meta", {})
    summary = result.get("diff_summary", {})
    changes = result.get("changes", [])
    taxonomy = result.get("risk_taxonomy_snapshot", {})

    total = summary.get("total_changes", len(changes))
    high = summary.get("medium_risk", 0)  # note: diff_summary uses these keys
    medium = summary.get("medium_risk", 0)
    low = summary.get("low_risk", 0)

    # Recompute from actual changes (more reliable than diff_summary)
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

    # Human verdict tracking (V2)
    h_confirmed = 0
    h_rejected = 0
    h_corrected = 0
    cat_corrections: dict[str, int] = {}
    level_shifts: dict[tuple[str, str], int] = {}
    correction_examples: list[dict] = []

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

        # ── Human verdict analysis (V2) ──────────────────────────
        hv = c.get("human_verdict")
        if not hv:
            continue
        action = hv.get("action", "")
        if action == "confirmed":
            h_confirmed += 1
        elif action == "rejected":
            h_rejected += 1
        elif action == "corrected":
            h_corrected += 1
            orig = hv.get("original", {})
            for cat in c.get("risk_categories", []) or []:
                cat_corrections[cat] = cat_corrections.get(cat, 0) + 1
            orig_lvl = orig.get("risk_level", "")
            new_lvl = c.get("risk_level", "")
            if orig_lvl and new_lvl and orig_lvl != new_lvl:
                key = (orig_lvl, new_lvl)
                level_shifts[key] = level_shifts.get(key, 0) + 1
            if len(correction_examples) < 10:
                correction_examples.append({
                    "change_id": c.get("id", ""),
                    "brief": (c.get("brief") or "")[:80],
                    "original_risk": orig.get("risk_level", "?"),
                    "corrected_risk": c.get("risk_level", "?"),
                    "original_categories": orig.get("risk_categories", []),
                    "corrected_categories": c.get("risk_categories", []),
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

    # Human feedback aggregation
    feedback_total = h_confirmed + h_rejected + h_corrected
    llm_error_rate = (h_rejected + h_corrected) / max(feedback_total, 1)

    most_corr = sorted(cat_corrections.items(), key=lambda x: x[1], reverse=True)[:5]
    most_corrected_categories = [
        {"id": cid, "name": _CAT_NAMES.get(cid, cid), "count": count}
        for cid, count in most_corr
    ]

    direction = sorted(level_shifts.items(), key=lambda x: x[1], reverse=True)[:5]
    correction_direction = [
        {"from": f, "to": t, "count": c}
        for (f, t), c in direction
    ]

    # Build summary string
    model_str = meta.get("model") or "offline"
    pipeline = meta.get("pipeline", "")
    parts = [f"{pipeline}({model_str}): {total}条差异"]
    if high:
        parts.append(f"{high}条高风险")
    if top_categories_out:
        tc = top_categories_out[0]
        parts.append(f"{tc['name']}{tc['count']}条")
    if h_corrected:
        parts.append(f"人工纠偏{h_corrected}条")
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
            "human_corrections_count": h_corrected,
            "high_confidence_verified_count": high_conf_verified,
        },
        "high_confidence_patterns": hc_patterns,
        "human_notes": human_notes[:20],
        "human_feedback": {
            "feedback_count": feedback_total,
            "feedback_rate": round(feedback_total / max(total, 1), 4),
            "confirmed_count": h_confirmed,
            "rejected_count": h_rejected,
            "corrected_count": h_corrected,
            "llm_error_rate": round(llm_error_rate, 4),
            "most_corrected_categories": most_corrected_categories,
            "correction_direction": correction_direction,
        },
        "correction_examples": correction_examples,
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
    hf = learning.get("human_feedback", {}) or {}
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
        "human_confirmed": hf.get("confirmed_count", 0),
        "human_corrected": hf.get("corrected_count", 0),
        "human_rejected": hf.get("rejected_count", 0),
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
        human_fb_total = 0
        human_err_total = 0
        human_err_rate_sum = 0.0
        human_err_rate_count = 0
        all_corrected_cats: dict[str, int] = {}
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
            # Human feedback aggregation
            hc = r.get("human_confirmed", 0) + r.get("human_corrected", 0) + r.get("human_rejected", 0)
            human_fb_total += hc
            herr = r.get("human_rejected", 0) + r.get("human_corrected", 0)
            human_err_total += herr
            if hc > 0:
                human_err_rate_sum += herr / hc
                human_err_rate_count += 1

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
            "total_human_feedback": human_fb_total,
            "avg_feedback_rate": round(human_fb_total / max(total_changes_sum, 1), 4),
            "avg_llm_error_rate": round(human_err_rate_sum / max(human_err_rate_count, 1), 4),
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

def load_past_learnings(limit: int = 5, full: bool = False,
                        data_dir: Optional[Path] = None) -> list[dict]:
    """Return the most recent run summaries from the index (top N).

    When full=True, loads complete per-run JSON files from
    data/learnings/run-{job_id}.json instead of just index summaries.
    Gracefully falls back to index entry if per-run file is missing.
    """
    d = _resolve_dir(data_dir)
    index = _load_index(d)
    run_entries = index.get("runs", [])[:limit]

    if not full:
        return run_entries

    full_records = []
    for entry in run_entries:
        run_path = d / f"run-{entry['job_id']}.json"
        if run_path.exists():
            try:
                with open(run_path, encoding="utf-8") as f:
                    full_records.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                full_records.append(entry)
        else:
            full_records.append(entry)
    return full_records


def format_learnings_context(learnings: list[dict]) -> str:
    """Format past learnings as a compact prompt context block.

    Includes both risk distribution history (V1) and human correction
    patterns (V2) when correction data is available.
    Handles both flat index entries and full learning records.
    """
    if not learnings:
        return ""

    def _get(l: dict, *keys: str, default=None):
        """Read from either flat index entry or nested learning record."""
        # Try flat key first (index entry)
        for k in keys:
            if k in l:
                return l[k]
        # Try nested risk_profile (full learning record)
        rp = l.get("risk_profile", {})
        if rp:
            for k in keys:
                if k in rp:
                    return rp[k]
        return default

    lines = [f"## 历史比对经验（基于 {len(learnings)} 次过往合同比对）", ""]
    lines.append("过往比对中常见的风险模式：")

    for i, l in enumerate(learnings, 1):
        ts = l.get("timestamp", "")[:16].replace("T", " ")
        model = l.get("model") or "offline"
        tc = l.get("top_category") or (
            (_get(l, "top_categories") or [{}])[0] if _get(l, "top_categories") else None
        )
        top_cat = f"{tc['name']}({tc['count']}条)" if tc else ""
        high = _get(l, "high_risk", "high_risk_count", default=0)
        medium = _get(l, "medium_risk", "medium_risk_count", default=0)
        lines.append(
            f"- 第{i}次({ts}): {top_cat} "
            f"高风险{high}条, 中风险{medium}条, "
            f"LLM校验拒绝率{_rate_pct(l.get('validation_rejection_rate'))}"
        )

    # Highlight top recurring categories across all runs
    all_cats: dict[str, int] = {}
    for l in learnings:
        # Try top_categories from index entry, or from nested risk_profile
        tcs = l.get("top_categories") or _get(l, "top_categories") or []
        for tc in tcs:
            all_cats[tc["name"]] = all_cats.get(tc["name"], 0) + tc["count"]
    top = sorted(all_cats.items(), key=lambda x: x[1], reverse=True)[:3]
    if top:
        lines.append("")
        lines.append(
            "重点关注: " + "、".join(f"{name}({total}次)" for name, total in top)
            + " 在过往比对中频繁出现。"
        )
        lines.append("在分析当前合同差异时，请优先检查这些高风险类别。")

    # ── Human correction patterns (V2) ────────────────────────────
    correction_examples: list[dict] = []
    for l in learnings:
        hf = l.get("human_feedback") or {}
        if hf.get("corrected_count", 0) > 0 or hf.get("rejected_count", 0) > 0:
            for ex in l.get("correction_examples", []) or []:
                if len(correction_examples) < 8:
                    correction_examples.append(ex)

    if correction_examples:
        lines.append("")
        lines.append("## 人工纠偏经验（基于历史人工审核）")
        lines.append("")
        lines.append("人工审核中发现的LLM常见误判模式，请在分析时特别注意避免重复：")
        lines.append("")

        cat_counts: dict[str, int] = {}
        for ex in correction_examples:
            for cat in ex.get("original_categories", []):
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_cats:
            lines.append("LLM常在这些类别上误判风险：")
            for cat_id, count in top_cats:
                lines.append(f"  - {_CAT_NAMES.get(cat_id, cat_id)}: {count}次被人工纠偏")
            lines.append("")

        lines.append("具体纠偏案例：")
        for ex in correction_examples[:5]:
            orig_cats = ", ".join(
                _CAT_NAMES.get(c, c) for c in ex.get("original_categories", [])
            )
            corr_cats = ", ".join(
                _CAT_NAMES.get(c, c) for c in ex.get("corrected_categories", [])
            )
            lines.append(
                f"  - [{ex.get('change_id', '?')}] {ex.get('brief', '')}: "
                f"LLM判为{ex.get('original_risk', '?')}风险({orig_cats}), "
                f"人工更正为{ex.get('corrected_risk', '?')}风险({corr_cats})"
            )

        lines.append("")
        lines.append("请谨慎评估上述类别的风险等级，避免重复LLM此前的误判模式。")

    return "\n".join(lines)


def _rate_pct(rate) -> str:
    if rate is None or rate == 0:
        return "N/A"
    return f"{rate * 100:.0f}%"
