#!/usr/bin/env python3
"""Contract Diff Agent

Usage:
    python -m src.main <v1_pdf> <v2_pdf> [--pipeline v04|v02] [--validate] [--offline] [-o result.json]
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

from .config import get_config
from .pipeline.extraction import extract_contract, estimate_tokens
from .pipeline.diff import diff_contracts


def _match_v2_page(snippet: str, pages) -> int | None:
    """Find which V2 page contains the given snippet text.

    Uses a tiered strategy: exact match → normalized whitespace → keyword overlap.
    Returns 1-indexed page number or None.
    """
    if not snippet or not pages:
        return None

    snippet = snippet.strip()
    if not snippet:
        return None

    # Normalize whitespace for matching
    def _norm(t):
        return "".join(t.split())

    norm_snippet = _norm(snippet)

    # Tier 1: exact substring match in page text
    for page in pages:
        if snippet in page.text:
            return page.page_num

    # Tier 2: exact match in normalized text
    for page in pages:
        if norm_snippet and norm_snippet in _norm(page.text):
            return page.page_num

    # Tier 3: match with key phrases (first 40+ chars if snippet is long enough)
    if len(norm_snippet) >= 40:
        key = norm_snippet[:40]
        for page in pages:
            if key in _norm(page.text):
                return page.page_num

    # Tier 4: match by longest common keyword sequence
    # Split into words/chars, find page with most keyword hits
    keywords = [w for w in norm_snippet if w.strip()]
    if len(keywords) >= 6:
        best_page = None
        best_score = 0
        for page in pages:
            norm_page = _norm(page.text)
            score = sum(1 for kw in keywords if kw in norm_page)
            if score > best_score:
                best_score = score
                best_page = page.page_num
        # Require at least 50% keyword overlap
        if best_score >= len(keywords) * 0.5:
            return best_page

    return None


def _run_v04(args, v1, v2, api_key: str, base_url: str, model: str,
             on_change: callable = None, on_progress: callable = None) -> dict:
    """v0.4 pipeline: traditional diff (always) → LLM enhance (optional) → validate → classify.

    Args:
        on_change: Optional callback(change_dict) for SSE streaming.
        on_progress: Optional callback(status, progress, step_label, message).
        args.offline: If True, skip all LLM calls.
        args.thorough: If True, separate diff enhance and risk classify.
    """
    from .pipeline.parsing import build_clause_tree
    from .pipeline.alignment import align_clauses
    from .pipeline.traditional_diff import traditional_diff

    def _progress(status, progress, step, msg=""):
        print(f"\n{msg or step}")
        if on_progress:
            try:
                on_progress(status, progress, step, msg)
            except Exception:
                pass

    offline = getattr(args, "offline", False) or not api_key
    thorough = getattr(args, "thorough", False)

    # ① 构建条款树
    _progress("tree_building", 12, "构建条款树", "正在解析条款结构...")
    tree1 = build_clause_tree(v1.full_text)
    tree2 = build_clause_tree(v2.full_text)
    print(f"   V1: {len(tree1.clauses)} L1 章节, {sum(len(c.children) for c in tree1.clauses)} L2 子条款")
    print(f"   V2: {len(tree2.clauses)} L1 章节, {sum(len(c.children) for c in tree2.clauses)} L2 子条款")

    # ② 条款对齐
    _progress("aligning", 20, "条款对齐", "匹配两个版本的条款结构...")
    diff_map = align_clauses(tree1, tree2)
    matched = sum(1 for p in diff_map.pairs if p.alignment_type == "match")
    print(f"   匹配: {matched} 对, 新增: {len(diff_map.v2_unmatched)}, 删除: {len(diff_map.v1_unmatched)}")

    # ③ 传统算法对比 (始终执行，离线可用)
    _progress("traditional_diff", 25, "传统算法对比", "逐条款文本对比，定位差异...")
    raw_changes = traditional_diff(diff_map)
    llm_used = False
    print(f"   传统算法识别到 {len(raw_changes)} 条差异")

    frequency: dict[str, int] = {}

    # ④ LLM 增强 (可选：仅在有 API key 且非离线模式时)
    if not offline:
        _progress("enhancing", 30, "LLM 增强描述", "AI 正在用自然语言描述变化...")
        try:
            from .pipeline.identifier import enhance_changes
            raw_changes, frequency = enhance_changes(
                raw_changes, diff_map, api_key=api_key, model=model, base_url=base_url,
                on_change=on_change,
            )
            llm_used = True
            print(f"   增强完成, {len(raw_changes)} 条变化已补充自然语言描述")

            # ⑤ thorough 模式：独立风险分类 (二次校验)
            if thorough:
                _progress("classifying", 75, "交叉校验风险分类", "独立模型二次校验，反幻觉...")
                from .pipeline.classifier import classify_changes
                raw_changes = classify_changes(
                    raw_changes, api_key=api_key, model=model, base_url=base_url,
                )
                frequency = {}
                for c in raw_changes:
                    for cat_id in c.get("risk_categories", []):
                        frequency[cat_id] = frequency.get(cat_id, 0) + 1
                print(f"   分类完成, {len(raw_changes)} 条变化已二次校验")
        except Exception as e:
            print(f"   ⚠️ LLM 增强失败: {e} — 返回传统算法结果")
            offline = True  # Fall back to offline
    else:
        print(f"   ⏭ 离线模式 — 跳过 LLM 增强")

    # ⑥ L2 校验 (始终执行，无 LLM 依赖)
    from .pipeline.validator import validate_changes as validate_l3
    from .pipeline.validator import _l2_check
    _progress("validating", 55, "L2 原文校验", "验证差异片段在原文中的存在性...")
    if not offline and args.validate:
        try:
            validated = validate_l3(
                raw_changes,
                v1.full_text, v2.full_text,
                api_key=api_key, model=model, base_url=base_url,
            )
        except Exception as e:
            print(f"   ⚠️ L3 校验失败: {e} — 仅保留 L2 结果")
            validated = []
            for c in raw_changes:
                val = {"validation": _l2_check(c, v1.full_text, v2.full_text)}
                val["validation"]["status"] = "l2_only"
                val["validation"]["l3_verdict"] = "skipped"
                c_with_val = dict(c)
                c_with_val["validation"] = val["validation"]
                validated.append(c_with_val)
    else:
        validated = []
        for c in raw_changes:
            val = {"validation": _l2_check(c, v1.full_text, v2.full_text)}
            val["validation"]["status"] = "l2_only"
            val["validation"]["l3_verdict"] = "skipped"
            c_with_val = dict(c)
            c_with_val["validation"] = val["validation"]
            validated.append(c_with_val)

    print(f"   校验完成: {len(validated)} 条")

    # ⑥.5 匹配 v2_page（在 v2.pages 中搜索每个 change 的 v2_snippet）
    for c in validated:
        c["v2_page"] = _match_v2_page(c.get("v2_snippet", ""), v2.pages)

    matched_pages = sum(1 for c in validated if c.get("v2_page"))
    print(f"   页码匹配: {matched_pages}/{len(validated)} 条定位到 V2 页码")

    # ⑦ 构建最终结果
    from .pipeline.classifier import save_taxonomy
    from .constants.risks import RISK_CATEGORIES
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

    pipeline_label = "offline" if offline else ("llm_enhanced" if llm_used else "traditional_only")

    result = {
        "meta": {
            "contract_v1": v1.file_path,
            "contract_v2": v2.file_path,
            "compared_at": datetime.now(timezone.utc).isoformat(),
            "agent_version": "1.4.0",
            "pipeline": f"v04-{pipeline_label}",
            "model": model if llm_used else None,
            "token_estimate": {
                "v1_tokens": estimate_tokens(v1.full_text),
                "v2_tokens": estimate_tokens(v2.full_text),
                "total": estimate_tokens(v1.full_text) + estimate_tokens(v2.full_text),
            },
        },
        "diff_summary": {
            "total_changes": len(validated),
            "confirmed": confirmed,
            "high_risk": sum(1 for c in validated if c.get("risk_level") == "high"),
            "medium_risk": sum(1 for c in validated if c.get("risk_level") == "medium"),
            "low_risk": sum(1 for c in validated if c.get("risk_level") == "low"),
            "alignment_coverage": matched / max(len(tree1.clauses) + len(tree2.clauses), 1),
            "llm_enhanced": llm_used,
        },
        "changes": validated,
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
    from .pipeline.diff import run_llm_diff
    return run_llm_diff(v1, v2, api_key=api_key, model=model, base_url=base_url)


def main():
    cfg = get_config()

    parser = argparse.ArgumentParser(
        description="Contract Diff Agent - Identify changes between two contract versions"
    )
    parser.add_argument("v1", help="Path to V1 contract PDF")
    parser.add_argument("v2", help="Path to V2 contract PDF")
    parser.add_argument("--pipeline", "-p", choices=["v02", "v03", "v04"], default="v04",
                        help="Pipeline: v04 (traditional+LLM, default), v03 (legacy LLM), v02 (full-text LLM)")
    parser.add_argument("--mode", "-m", choices=["text", "llm"],
                        help="[deprecated] Use --pipeline instead")
    parser.add_argument("--model", default=cfg.model,
                        help=f"Model for LLM calls (default: {cfg.model})")
    parser.add_argument("--validate", "-V", action="store_true",
                        help="Enable L3 LLM semantic validation (slower but catches hallucinations)")
    parser.add_argument("--thorough", action="store_true",
                        help="Separate diff and risk classification for cross-validation (slower, higher quality)")
    parser.add_argument("--offline", "-O", action="store_true",
                        help="Skip all LLM calls — produce traditional diff only (works without API key)")
    parser.add_argument("--bilingual", action="store_true",
                        help="Keep bilingual content (English + Chinese)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")

    args = parser.parse_args()

    # Backward compat
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

    api_key = cfg.api_key
    base_url = cfg.base_url

    # Route to pipeline
    if args.pipeline == "text" or args.mode == "text":
        print(f"\n🔍 文本 Diff (v0.1)...")
        result = diff_contracts(v1, v2)
    elif args.pipeline == "v02":
        if not api_key and not args.offline:
            print("❌ v02 需要 API Key (请设置 LLM_API_KEY 环境变量，或用 --offline 降级到 v04)")
            sys.exit(1)
        print(f"\n🔍 LLM 全文比对 (v0.2)...")
        result = _run_v02(args, v1, v2, api_key, base_url, args.model)
    elif args.pipeline == "v03":
        if not api_key and not args.offline:
            print("❌ v03 需要 API Key (请设置 LLM_API_KEY 环境变量，或用 --offline 降级到 v04)")
            sys.exit(1)
        from .pipeline.parsing import build_clause_tree
        from .pipeline.alignment import align_clauses
        from .pipeline.identifier import identify_changes
        print(f"\n🔍 条款树比对 v0.3 (LLM 原生识别)...")
        tree1 = build_clause_tree(v1.full_text)
        tree2 = build_clause_tree(v2.full_text)
        diff_map = align_clauses(tree1, tree2)
        raw_changes, _ = identify_changes(diff_map, api_key=api_key, model=args.model, base_url=base_url)
        result = {
            "meta": {
                "contract_v1": v1.file_path, "contract_v2": v2.file_path,
                "compared_at": datetime.now(timezone.utc).isoformat(),
                "agent_version": "0.3.0", "pipeline": "clause-tree-legacy",
                "model": args.model,
            },
            "diff_summary": {"total_changes": len(raw_changes), "confirmed": len(raw_changes)},
            "changes": raw_changes,
            "risk_taxonomy_snapshot": {},
            "unmatched_content": {},
        }
    else:
        # v04 (default): traditional diff + optional LLM enhancement
        mode_str = "离线模式" if (args.offline or not api_key) else "传统+LLM增强"
        print(f"\n🔍 条款树比对 v0.4 ({mode_str})...")
        result = _run_v04(args, v1, v2, api_key, base_url, args.model)

    # Self-evolution: extract and save learning (non-fatal)
    try:
        from src.pipeline.learning import extract_learning, save_learning
        import hashlib
        from datetime import datetime, timezone
        raw_id = f"{args.v1}{args.v2}{datetime.now(timezone.utc).isoformat()}"
        jid = hashlib.md5(raw_id.encode()).hexdigest()[:8]
        learning = extract_learning(result, jid)
        save_learning(learning)
    except Exception as e:
        print(f"  ⚠️ 进化记录保存失败: {e}")

    # Display results
    print(f"\n--- 比对结果 ---")
    s = result["diff_summary"]
    print(f"总变更: {s['total_changes']} | "
          f"高风险: {s.get('high_risk', '?')} | "
          f"中风险: {s.get('medium_risk', '?')} | "
          f"低风险: {s.get('low_risk', '?')}")

    freq = result.get("risk_taxonomy_snapshot", {}).get("frequency", {})
    if freq:
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"高频风险: {', '.join(f'{k}({v})' for k, v in top)}")

    for change in result["changes"][:10]:
        risk_lvl = change.get("risk_level", "?")
        note = change.get("risk_note", "")[:80]
        print(f"  [{change['change_type'].upper()}][{risk_lvl}] {change.get('brief', '')[:80]}")
        if note:
            print(f"    ⚠ {note}")

    if len(result["changes"]) > 10:
        print(f"  ... ({len(result['changes']) - 10} more changes)")

    # Write output
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    out_path = cfg.resolve_output(args.output)
    out_path.write_text(json_str, encoding="utf-8")
    print(f"\n✅ 结果已写入: {out_path}")


if __name__ == "__main__":
    main()
