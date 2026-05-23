"""Diff content identifier - Step ③+⑤ of the contract diff pipeline.

Takes aligned clause pairs and uses LLM to identify semantic differences
AND classify risks in a single call. Parallelized for speed.
"""

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from json_repair import repair_json

from .clause_aligner import DiffMap, AlignedPair
from .clause_tree import ClauseNode
from .pdf_extractor import estimate_tokens
from .risk_classifier import RISK_CATEGORIES
from .model_pool import AutoFallbackClient

_MAX_WORKERS = 5
_COUNTER_LOCK = threading.Lock()

IDENTIFIER_SYSTEM = """你是一名合同比对专家。对比同一合同条款的两个版本，找出实质性变化并做风险分类。

## 风险分类标准
{risk_categories}

## 风险等级
- high: 涉及金额、权责、时效的实质性变化
- medium: 措辞调整可能影响权利义务解释
- low: 格式、编号、非实质性文字调整

## 输出 JSON
{{
  "changes": [
    {{
      "id": "change-1",
      "change_type": "added|removed|modified",
      "brief": "一句话摘要（30字以内，通俗语言）",
      "v1_snippet": "V1原文关键句或null",
      "v2_snippet": "V2原文关键句或null",
      "confidence": "high|medium|low",
      "risk_categories": ["R01"],
      "risk_level": "high|medium|low",
      "risk_note": "通俗风险提示（50字以内）",
      "attention_for": "建议关注部门或null",
      "is_favorable": true/false/null
    }}
  ],
  "no_change": false
}}

is_favorable: 对我方（商户/甲方）有利填true, 不利填false, 无法判断填null
risk_note: 面向非法律人员，通俗易懂"""

IDENTIFIER_USER = """## {clause_title}
## V1 ({v1_ref}):
{v1_text}

## V2 ({v2_ref}):
{v2_text}

输出 JSON。"""


def _build_pair_prompt(pair: AlignedPair) -> tuple[str, str] | None:
    c1 = pair.v1_clause
    c2 = pair.v2_clause
    if not c1 and not c2:
        return None

    title = c1.title if c1 else (c2.title if c2 else "未知")
    v1_ref = f"{c1.number}、{c1.title}" if c1 else "不存在"
    v2_ref = f"{c2.number}、{c2.title}" if c2 else "不存在"
    v1_text = c1.full_text if c1 else "（V1 中无此条款）"
    v2_text = c2.full_text if c2 else "（V2 中已删除此条款）"

    if (not c1 or not c1.full_text.strip()) and (not c2 or not c2.full_text.strip()):
        return None

    return title, IDENTIFIER_USER.format(
        clause_title=title,
        v1_ref=v1_ref,
        v2_ref=v2_ref,
        v1_text=v1_text[:3000],
        v2_text=v2_text[:3000],
    )


def _build_system_prompt() -> str:
    cat_lines = "\n".join(
        f"- {c['id']}: {c['name']}（关注：{c['focus']}）"
        for c in RISK_CATEGORIES
    )
    return IDENTIFIER_SYSTEM.format(risk_categories=cat_lines)


def _prefilter(pair: AlignedPair) -> list[dict] | None:
    """Check if clause texts are nearly identical → skip LLM. Returns None if LLM needed."""
    c1 = pair.v1_clause
    c2 = pair.v2_clause
    if not c1 or not c2:
        return None  # Added/removed → always needs LLM

    t1 = c1.full_text.strip()
    t2 = c2.full_text.strip()

    if not t1 and not t2:
        return []  # Both empty → no change

    if not t1 or not t2:
        return None  # One side empty → needs LLM

    # Strip whitespace for comparison
    clean1 = re.sub(r'\s+', '', t1)
    clean2 = re.sub(r'\s+', '', t2)

    if clean1 == clean2:
        return []  # Identical after whitespace normalization → truly no change

    return None  # Any difference → must go through LLM


def _diff_single(
    pair: AlignedPair,
    client: AutoFallbackClient,
    max_tokens: int,
    system_prompt: str,
    pair_index: int,
    total_pairs: int,
) -> list[dict]:
    """Diff a single clause pair, returning changes with risk classification."""
    # Pre-filter: skip LLM for nearly identical texts
    pre = _prefilter(pair)
    if pre is not None:
        if not pre:
            print(f"  [{pair_index}/{total_pairs}] {pair.v1_clause.title if pair.v1_clause else '?'}... → 0 changes (pre-filtered)")
        return pre

    prompt_info = _build_pair_prompt(pair)
    if prompt_info is None:
        return []

    title, user_prompt = prompt_info

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
        if raw.get("no_change"):
            return []

        changes = _normalize_changes(raw.get("changes", []))
        print(f"  [{pair_index}/{total_pairs}] {title[:30]}... → {len(changes)} changes")
        return changes

    except Exception as e:
        print(f"  ⚠️ [{pair_index}/{total_pairs}] {title[:30]}... 失败: {e}")
        return []


def _normalize_changes(raw_changes: list[dict]) -> list[dict]:
    """Normalize LLM output fields to our standard schema."""
    normalized = []
    for c in raw_changes:
        nc = {
            "change_type": _infer_change_type(c),
            "brief": c.get("brief") or c.get("description") or c.get("summary", ""),
            "v1_snippet": c.get("v1_snippet") or c.get("v1", ""),
            "v2_snippet": c.get("v2_snippet") or c.get("v2", ""),
            "confidence": c.get("confidence", "medium"),
            "risk_categories": c.get("risk_categories", []),
            "risk_level": c.get("risk_level", "medium"),
            "risk_note": c.get("risk_note", ""),
            "attention_for": c.get("attention_for"),
            "is_favorable": c.get("is_favorable"),
            "source": "llm",
        }
        for key in list(nc.keys()):
            if nc[key] is None or nc[key] == "":
                del nc[key]
        normalized.append(nc)
    return normalized


def _infer_change_type(c: dict) -> str:
    raw_type = c.get("change_type") or c.get("type") or c.get("category", "")
    if "新增" in raw_type or "add" in raw_type.lower():
        return "added"
    if "删除" in raw_type or "remov" in raw_type.lower():
        return "removed"
    if "修改" in raw_type or "变更" in raw_type or "modif" in raw_type.lower():
        return "modified"
    if not c.get("v1") and c.get("v2"):
        return "added"
    if c.get("v1") and not c.get("v2"):
        return "removed"
    return "modified"


def identify_changes(
    diff_map: DiffMap,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4.6",
    base_url: str = "https://api.gmi-serving.com/v1",
    max_tokens: int = 2000,
    max_workers: int = _MAX_WORKERS,
    on_change: callable = None,
) -> tuple[list[dict], dict[str, int]]:
    """Run LLM diff + risk classification on aligned pairs IN PARALLEL.

    Args:
        on_change: Optional callback(change_dict) called for each new change.
                   Used for SSE streaming in the web UI.

    Returns (changes, frequency_dict).
    """
    system_prompt = _build_system_prompt()
    frequency: dict[str, int] = {}

    # Collect all pairs to process
    tasks: list[AlignedPair] = []
    l1_pairs = [p for p in diff_map.pairs if p.v1_clause and p.v2_clause
                and p.v1_clause.level == 1]

    for l1_pair in l1_pairs:
        tasks.append(l1_pair)
        # Add L2 children
        for p in diff_map.pairs:
            if (p.v1_clause and p.v2_clause and p.v1_clause.level == 2
                    and p.v1_clause.id.startswith(f"{l1_pair.v1_clause.id}.")):
                tasks.append(p)

    # Add added/removed (algorithmic, no LLM needed)
    algo_changes: list[dict] = []
    for p in diff_map.pairs:
        if p.alignment_type == "added" and p.v1_clause is None:
            c2 = p.v2_clause
            algo_changes.append({
                "change_type": "added",
                "clause_ref_v1": None,
                "clause_ref_v2": f"{c2.number} {c2.title}" if c2 else None,
                "brief": f"新增条款: {c2.title if c2 else ''}",
                "v2_snippet": c2.full_text[:200] if c2 else None,
                "confidence": "high", "risk_categories": [],
                "risk_level": "medium", "risk_note": "",
                "source": "algorithm",
            })
        elif p.alignment_type == "removed" and p.v2_clause is None:
            c1 = p.v1_clause
            algo_changes.append({
                "change_type": "removed",
                "clause_ref_v1": f"{c1.number} {c1.title}" if c1 else None,
                "clause_ref_v2": None,
                "brief": f"删除条款: {c1.title if c1 else ''}",
                "v1_snippet": c1.full_text[:200] if c1 else None,
                "confidence": "high", "risk_categories": [],
                "risk_level": "medium", "risk_note": "",
                "source": "algorithm",
            })

    total = len(tasks)
    print(f"  共 {total} 个条款对, {max_workers} 路并行处理...")

    all_changes: list[dict] = list(algo_changes)
    counter = len(algo_changes)

    if not tasks:
        return all_changes, frequency

    # Parallel execution with auto-fallback across models
    client = AutoFallbackClient(api_key=api_key, base_url=base_url,
                                primary_model=model, timeout=300.0)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _diff_single, pair, client, max_tokens, system_prompt, i, total
            ): (i, pair)
            for i, pair in enumerate(tasks, 1)
        }

        for future in as_completed(futures):
            changes = future.result()
            completed += 1

            with _COUNTER_LOCK:
                for change in changes:
                    counter += 1
                    change["id"] = f"diff-{counter:03d}"
                    for cat_id in change.get("risk_categories", []):
                        frequency[cat_id] = frequency.get(cat_id, 0) + 1
                    if on_change:
                        try:
                            on_change(dict(change))
                        except Exception:
                            pass
                all_changes.extend(changes)

    # Set clause refs from pair context
    for change in all_changes:
        if "clause_ref_v1" not in change:
            change["clause_ref_v1"] = None
        if "clause_ref_v2" not in change:
            change["clause_ref_v2"] = None

    return all_changes, frequency
