#!/usr/bin/env python3
"""Contract Diff Agent

Usage:
    python -m src.main <v1_pdf> <v2_pdf> [--pipeline v03|v02] [--validate] [-o result.json]
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .pdf_extractor import extract_contract, estimate_tokens
from .diff_engine import diff_contracts


def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _run_v03(args, v1, v2, api_key: str, base_url: str, model: str,
             on_change: callable = None, on_progress: callable = None) -> dict:
    """Full pipeline: clause tree → align → identify → validate → classify.

    Args:
        on_change: Optional callback(change_dict) for SSE streaming.
        on_progress: Optional callback(status, progress, step_label, message).
        args.thorough: If True, separate ③ diff and ⑤ risk for cross-validation.
    """
    from .clause_tree import build_clause_tree
    from .clause_aligner import align_clauses
    from .diff_identifier import identify_changes

    def _progress(status, progress, step, msg=""):
        print(f"\n{msg or step}")
        if on_progress:
            try:
                on_progress(status, progress, step, msg)
            except Exception:
                pass

    _progress("tree_building", 12, "构建条款树", "正在解析条款结构...")
    tree1 = build_clause_tree(v1.full_text)
    tree2 = build_clause_tree(v2.full_text)
    print(f"   V1: {len(tree1.clauses)} L1 章节, {sum(len(c.children) for c in tree1.clauses)} L2 子条款")
    print(f"   V2: {len(tree2.clauses)} L1 章节, {sum(len(c.children) for c in tree2.clauses)} L2 子条款")

    _progress("aligning", 20, "条款对齐", "匹配两个版本的条款结构...")
    diff_map = align_clauses(tree1, tree2)
    matched = sum(1 for p in diff_map.pairs if p.alignment_type == "match")
    print(f"   匹配: {matched} 对, 新增: {len(diff_map.v2_unmatched)}, 删除: {len(diff_map.v1_unmatched)}")

    thorough = getattr(args, "thorough", False)

    _progress("identifying", 30, "LLM 逐条比对差异", "AI 正在对比两个版本的条款内容...")
    if thorough:
        raw_changes, _ = identify_changes(
            diff_map, api_key=api_key, model=model, base_url=base_url,
            on_change=on_change, skip_risk=True,
        )
        print(f"   识别到 {len(raw_changes)} 条变化")

        print(f"\n⑤ 独立风险分类 (严谨模式: 二次校验)...")
        from .risk_classifier import classify_changes
        raw_changes = classify_changes(
            raw_changes, api_key=api_key, model=model, base_url=base_url,
        )
        # Recompute frequency from classified changes
        frequency: dict[str, int] = {}
        for c in raw_changes:
            for cat_id in c.get("risk_categories", []):
                frequency[cat_id] = frequency.get(cat_id, 0) + 1
        print(f"   分类完成, {len(raw_changes)} 条变化已二次校验")
    else:
        print(f"\n③ LLM 差异识别 + 风险分类 (并行)...")
        raw_changes, frequency = identify_changes(
            diff_map, api_key=api_key, model=model, base_url=base_url,
            on_change=on_change,
        )
        print(f"   识别到 {len(raw_changes)} 条变化(含风险分类)")

    # L2 validation (fast, always runs)
    from .validator import validate_changes as validate_l3
    from .validator import _l2_check
    _progress("validating", 55, "校验差异", "正在验证差异真实性...")
    if args.validate:
        validated = validate_l3(
            raw_changes,
            v1.full_text, v2.full_text,
            api_key=api_key, model=model, base_url=base_url,
        )
    else:
        validated = []
        for c in raw_changes:
            val = {"validation": _l2_check(c, v1.full_text, v2.full_text)}
            val["validation"]["status"] = "l2_only"
            val["validation"]["l3_verdict"] = "skipped"
            c_with_val = dict(c)
            c_with_val["validation"] = val["validation"]
            validated.append(c_with_val)

    # Compute risk taxonomy snapshot
    from .risk_classifier import save_taxonomy, RISK_CATEGORIES
    categories_used: set[str] = set()
    for c in validated:
        for cat_id in c.get("risk_categories", []):
            categories_used.add(cat_id)

    high_freq = [k for k, v in frequency.items() if v >= 10]
    confirmed = sum(1 for vc in validated
                    if vc.get("validation", {}).get("l3_verdict") == "confirmed"
                    or vc.get("validation", {}).get("status", "") in ("l2_only", "skipped")
                    or vc.get("validation", {}).get("status") is None)

    save_taxonomy(frequency, [])
    cat_names = {c["id"]: c["name"] for c in RISK_CATEGORIES}

    classified = validated  # alias for result building

    # Build final result
    result = {
        "meta": {
            "contract_v1": v1.file_path,
            "contract_v2": v2.file_path,
            "compared_at": datetime.now(timezone.utc).isoformat(),
            "agent_version": "0.4.0",
            "pipeline": "clause-tree",
            "model": model,
            "token_estimate": {
                "v1_tokens": estimate_tokens(v1.full_text),
                "v2_tokens": estimate_tokens(v2.full_text),
                "total": estimate_tokens(v1.full_text) + estimate_tokens(v2.full_text),
            },
        },
        "diff_summary": {
            "total_changes": len(classified),
            "confirmed": confirmed,
            "high_risk": sum(1 for c in classified if c.get("risk_level") == "high"),
            "medium_risk": sum(1 for c in classified if c.get("risk_level") == "medium"),
            "low_risk": sum(1 for c in classified if c.get("risk_level") == "low"),
            "alignment_coverage": matched / max(len(tree1.clauses) + len(tree2.clauses), 1),
        },
        "changes": classified,
        "risk_taxonomy_snapshot": {
            "categories_used": sorted(categories_used),
            "frequency": {cat_names.get(k, k): v for k, v in frequency.items()},
            "high_frequency_alerts": [cat_names.get(k, k) for k in high_freq],
        },
        "unmatched_content": {
            "v1_only": [f"{c.number}、{c.title}" for c in diff_map.v1_unmatched],
            "v2_only": [f"{c.number}、{c.title}" for c in diff_map.v2_unmatched],
            "note": "These chapters could not be automatically aligned",
        },
    }

    return result


def _run_v02(args, v1, v2, api_key: str, base_url: str, model: str) -> dict:
    """Old v0.2 pipeline: full-text LLM diff."""
    from .llm_diff import run_llm_diff
    return run_llm_diff(v1, v2, api_key=api_key, model=model, base_url=base_url)


def main():
    _load_env()
    parser = argparse.ArgumentParser(
        description="Contract Diff Agent - Identify changes between two contract versions"
    )
    parser.add_argument("v1", help="Path to V1 contract PDF")
    parser.add_argument("v2", help="Path to V2 contract PDF")
    parser.add_argument("--pipeline", "-p", choices=["v02", "v03"], default="v03",
                        help="Pipeline version: v03 (clause-tree, default) or v02 (full-text LLM)")
    parser.add_argument("--mode", "-m", choices=["text", "llm"],
                        help="[deprecated] Use --pipeline instead")
    parser.add_argument("--model", default=os.environ.get("CLAUDE_MODEL", "anthropic/claude-sonnet-4.6"),
                        help="Claude model for LLM calls")
    parser.add_argument("--validate", "-V", action="store_true",
                        help="Enable L3 LLM semantic validation (slower but catches hallucinations)")
    parser.add_argument("--thorough", action="store_true",
                        help="Separate diff and risk classification for cross-validation (slower, higher quality)")
    parser.add_argument("--bilingual", action="store_true",
                        help="Keep bilingual content (English + Chinese)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")

    args = parser.parse_args()

    # Backward compat: --mode text
    if args.mode == "text":
        args.pipeline = "text"

    print(f"📄 提取 V1: {args.v1}")
    v1 = extract_contract(args.v1, keep_english=args.bilingual)

    print(f"📄 提取 V2: {args.v2}")
    v2 = extract_contract(args.v2, keep_english=args.bilingual)

    print(f"\n--- 文档概要 ---")
    print(f"V1: {v1.total_pages} 页, ~{estimate_tokens(v1.full_text):,} tokens")
    print(f"V2: {v2.total_pages} 页, ~{estimate_tokens(v2.full_text):,} tokens")
    print(f"总计: ~{estimate_tokens(v1.full_text) + estimate_tokens(v2.full_text):,} tokens")

    if not args.bilingual:
        saved = estimate_tokens(v2.full_text_bilingual) - estimate_tokens(v2.full_text)
        if saved > 0:
            print(f"英文内容已过滤, 节省 ~{saved:,} tokens")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("GMI_BASE_URL", "https://api.gmi-serving.com/v1")

    # Route to pipeline
    if args.pipeline == "text" or args.mode == "text":
        print(f"\n🔍 文本 Diff (v0.1)...")
        result = diff_contracts(v1, v2)
    elif args.pipeline == "v02":
        if not api_key:
            print("❌ 请设置 ANTHROPIC_API_KEY")
            sys.exit(1)
        print(f"\n🔍 LLM 全文比对 (v0.2)...")
        result = _run_v02(args, v1, v2, api_key, base_url, args.model)
    else:
        if not api_key:
            print("❌ 请设置 ANTHROPIC_API_KEY")
            sys.exit(1)
        validate_str = "含L3校验" if args.validate else "L2校验"
        print(f"\n🔍 条款树比对 v0.3 ({validate_str})...")
        result = _run_v03(args, v1, v2, api_key, base_url, args.model)

    # Display results
    print(f"\n--- 比对结果 ---")
    s = result["diff_summary"]
    print(f"总变更: {s['total_changes']} | "
          f"高风险: {s.get('high_risk', '?')} | "
          f"中风险: {s.get('medium_risk', '?')} | "
          f"低风险: {s.get('low_risk', '?')}")

    # Top risk categories
    freq = result.get("risk_taxonomy_snapshot", {}).get("frequency", {})
    if freq:
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"高频风险: {', '.join(f'{k}({v})' for k, v in top)}")

    for change in result["changes"][:10]:
        risk_lvl = change.get("risk_level", "?")
        cats = ",".join(change.get("risk_categories", []))
        note = change.get("risk_note", "")[:80]
        print(f"  [{change['change_type'].upper()}][{risk_lvl}] {change.get('brief', '')[:80]}")
        if note:
            print(f"    ⚠ {note}")

    if len(result["changes"]) > 10:
        print(f"  ... ({len(result['changes']) - 10} more changes)")

    # Write output
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    out_path = args.output or "data/diff_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"\n✅ 结果已写入: {out_path}")


if __name__ == "__main__":
    main()
