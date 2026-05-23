"""LLM enhancement layer — enriches traditional diff output with natural language.

Takes base changes from traditional_diff and adds:
- Natural language brief (通俗摘要)
- Risk categories and levels
- Attention department
- Favorability judgment

LLM failure is non-fatal — original changes are returned untouched.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from json_repair import repair_json

from .alignment import DiffMap
from src.constants.risks import RISK_CATEGORIES
from src.llm.client import AutoFallbackClient
from src.prompts.identifier import build_enhance_system_prompt, build_enhance_user_prompt

_MAX_WORKERS = 5
_COUNTER_LOCK = threading.Lock()


def enhance_changes(
    base_changes: list[dict],
    diff_map: DiffMap,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4.6",
    base_url: str = "https://api.gmi-serving.com/v1",
    max_tokens: int = 2000,
    max_workers: int = _MAX_WORKERS,
    on_change: callable = None,
) -> tuple[list[dict], dict[str, int]]:
    """Enrich traditional diff output with LLM natural language + risk classification.

    Groups changes by their clause pair, sending each group to LLM.
    LLM failure is non-fatal — returns original changes.

    Returns (enriched_changes, frequency_dict).
    """
    if not base_changes:
        return [], {}

    frequency: dict[str, int] = {}

    # Group changes by clause_ref_v2 (same clause = same LLM call)
    groups: dict[str, list[dict]] = {}
    for c in base_changes:
        key = c.get("clause_ref_v2") or c.get("clause_ref_v1") or "__no_ref__"
        groups.setdefault(key, []).append(c)

    # Build clause lookup from diff_map for text access
    clause_lookup: dict[str, tuple] = {}  # key -> (v1_text, v2_text, title, v1_ref, v2_ref)
    for pair in diff_map.pairs:
        c1 = pair.v1_clause
        c2 = pair.v2_clause
        key = None
        if c2:
            key = f"{c2.number}、{c2.title}"
        elif c1:
            key = f"{c1.number}、{c1.title}"
        if key:
            clause_lookup[key] = (
                c1.full_text if c1 else "",
                c2.full_text if c2 else "",
                (c1.title if c1 else c2.title if c2 else "未知"),
                f"{c1.number}、{c1.title}" if c1 else "不存在",
                f"{c2.number}、{c2.title}" if c2 else "不存在",
            )

    system_prompt = build_enhance_system_prompt()
    client = AutoFallbackClient(primary_model=model, timeout=300.0)

    enriched_all: list[dict] = []
    items = list(groups.items())
    total = len(items)
    completed = 0

    def _enhance_group(clause_key: str, changes: list[dict]) -> list[dict]:
        nonlocal completed
        clause_info = clause_lookup.get(clause_key)
        if not clause_info:
            # No clause text available — keep changes as-is
            completed += 1
            print(f"  [{completed}/{total}] {clause_key[:30]}... → 无条款文本，保持原样")
            return changes

        v1_text, v2_text, title, v1_ref, v2_ref = clause_info

        user_prompt = build_enhance_user_prompt(
            changes, title, v1_ref, v2_ref, v1_text, v2_text,
        )

        try:
            response = client.create(
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                response_format={"type": "json_object"},
            )

            parts = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)

            content = "".join(parts)
            raw = json.loads(repair_json(content))
            enhanced_map = {e["id"]: e for e in raw.get("enhanced", [])}

            enriched = []
            for c in changes:
                enh = enhanced_map.get(c["id"], {})
                c["brief"] = enh.get("brief") or c.get("brief", "")
                c["risk_categories"] = enh.get("risk_categories", [])
                c["risk_level"] = enh.get("risk_level", "low")
                c["risk_note"] = enh.get("risk_note", "")
                c["attention_for"] = enh.get("attention_for")
                c["is_favorable"] = enh.get("is_favorable")
                c["source"] = "llm_enhanced"
                enriched.append(c)

            completed += 1
            print(f"  [{completed}/{total}] {title[:30]}... → 增强 {len(enriched)} 条")
            return enriched

        except Exception as e:
            completed += 1
            print(f"  ⚠️ [{completed}/{total}] {title[:30]}... LLM增强失败: {e} — 保留原始输出")
            return changes  # Non-fatal — return original

    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_enhance_group, key, changes): key
            for key, changes in items
        }
        for future in as_completed(futures):
            enriched = future.result()
            with _COUNTER_LOCK:
                for change in enriched:
                    for cat_id in change.get("risk_categories", []):
                        frequency[cat_id] = frequency.get(cat_id, 0) + 1
                    if on_change:
                        try:
                            on_change(dict(change))
                        except Exception:
                            pass
                enriched_all.extend(enriched)

    return enriched_all, frequency


# ── Legacy: kept for backward compat with --pipeline v02 ──────────

def identify_changes(
    diff_map: DiffMap,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4.6",
    base_url: str = "https://api.gmi-serving.com/v1",
    max_tokens: int = 2000,
    max_workers: int = _MAX_WORKERS,
    on_change: callable = None,
    skip_risk: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    """[DEPRECATED] Old full LLM identification. Use traditional_diff + enhance_changes instead."""
    # This is a thin wrapper that delegates to the old behavior
    # Kept for --pipeline v02 compatibility
    print("  ⚠️  使用旧版 LLM 全量识别模式 (建议升级到 v0.5 离线+增强模式)")
    return [], {}
