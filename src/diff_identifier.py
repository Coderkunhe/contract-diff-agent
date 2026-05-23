"""Diff content identifier - Step ③ of the contract diff pipeline.

Takes aligned clause pairs and uses LLM to identify semantic differences.
Each aligned pair is diff'd independently → parallelizable.
"""

import json
import re
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI
from json_repair import repair_json

from .clause_aligner import DiffMap, AlignedPair
from .clause_tree import ClauseNode
from .pdf_extractor import estimate_tokens

IDENTIFIER_SYSTEM = """你是一名合同比对专家。你的任务是对比同一合同条款的两个版本，
找出内容层面的实质性变化。

## 输入格式
你会收到来自 V1（旧版）和 V2（新版）的对应条款文本。可能来自完全对齐的条款对，
也可能来自仅在一版中存在的条款。

## 输出要求

1. 找出所有实质性内容变化（忽略中英文翻译差异、标点符号调整）
2. 变化描述要精准，用通俗语言
3. 对每条变化评估置信度：
   - high: 原文明确支持该变化
   - medium: 原文部分支持，存在一定解读空间
   - low: 变化判断主要基于推理，原文仅有间接线索

## 输出格式

直接输出一个 JSON 对象：
{
  "changes": [
    {
      "id": "change-1",
      "change_type": "added|removed|modified",
      "brief": "一句话摘要（30字以内）",
      "v1_snippet": "V1原文关键句或null",
      "v2_snippet": "V2原文关键句或null",
      "confidence": "high|medium|low"
    }
  ],
  "no_change": false
}

如果两份条款没有实质差异，输出 {"no_change": true, "changes": []}"""

IDENTIFIER_USER = """## 条款: {clause_title}
## V1 ({v1_ref}):
{v1_text}

## V2 ({v2_ref}):
{v2_text}

请找出以上条款的实质性内容变化，输出 JSON。"""


def _build_pair_prompt(pair: AlignedPair) -> tuple[str, str] | None:
    """Build a prompt for a single aligned pair. Returns (clause_title, prompt) or None."""
    c1 = pair.v1_clause
    c2 = pair.v2_clause

    title = c1.title if c1 else (c2.title if c2 else "未知")
    v1_ref = f"{c1.number}、{c1.title}" if c1 else "不存在"
    v2_ref = f"{c2.number}、{c2.title}" if c2 else "不存在"
    v1_text = c1.full_text if c1 else "（V1 中无此条款）"
    v2_text = c2.full_text if c2 else "（V2 中已删除此条款）"

    # Skip if both sides are empty
    if (not c1 or not c1.full_text.strip()) and (not c2 or not c2.full_text.strip()):
        return None

    if not v1_text.strip() and not v2_text.strip():
        return None

    return title, IDENTIFIER_USER.format(
        clause_title=title,
        v1_ref=v1_ref,
        v2_ref=v2_ref,
        v1_text=v1_text[:3000],
        v2_text=v2_text[:3000],
    )


def identify_changes(
    diff_map: DiffMap,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4.6",
    base_url: str = "https://api.gmi-serving.com/v1",
    max_tokens: int = 2500,
) -> list[dict]:
    """Run LLM-based diff identification on aligned clause pairs.

    Returns a list of change dicts across all pairs.
    """
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
    all_changes: list[dict] = []
    change_counter = 0

    # Process matched L1 pairs + their L2 children
    l1_pairs = [p for p in diff_map.pairs if p.v1_clause and p.v2_clause
                and p.v1_clause.level == 1]

    print(f"  共 {len(l1_pairs)} 个对齐章节, 逐个识别差异...")

    for pair in l1_pairs:
        # Diff the L1 bodies (text between title and first sub-clause)
        result = _diff_pair(client, pair, model, max_tokens)
        if result:
            for change in result.get("changes", []):
                change_counter += 1
                change["id"] = f"diff-{change_counter:03d}"
                change["clause_ref_v1"] = f"{pair.v1_clause.number}、{pair.v1_clause.title}"
                change["clause_ref_v2"] = f"{pair.v2_clause.number}、{pair.v2_clause.title}"
                change["source"] = "llm"
            all_changes.extend(result.get("changes", []))

        # Diff matched L2 children within this L1
        l2_pairs = [p for p in diff_map.pairs if p.v1_clause and p.v2_clause
                    and p.v1_clause.level == 2
                    and p.v1_clause.id.startswith(f"{pair.v1_clause.id}.")]

        for l2_pair in l2_pairs:
            result = _diff_pair(client, l2_pair, model, max_tokens)
            if result:
                for change in result.get("changes", []):
                    change_counter += 1
                    change["id"] = f"diff-{change_counter:03d}"
                    change["clause_ref_v1"] = f"{l2_pair.v1_clause.number} {l2_pair.v1_clause.title}"
                    change["clause_ref_v2"] = f"{l2_pair.v2_clause.number} {l2_pair.v2_clause.title}"
                    change["source"] = "llm"
                all_changes.extend(result.get("changes", []))

    # Process added/removed clauses (entirely new or deleted)
    for pair in diff_map.pairs:
        if pair.alignment_type in ("added", "removed") and pair.v1_clause is None:
            change_counter += 1
            c2 = pair.v2_clause
            all_changes.append({
                "id": f"diff-{change_counter:03d}",
                "change_type": "added",
                "clause_ref_v1": None,
                "clause_ref_v2": f"{c2.number} {c2.title}" if c2 else None,
                "brief": f"新增条款: {c2.title if c2 else ''}",
                "v1_snippet": None,
                "v2_snippet": c2.full_text[:200] if c2 else None,
                "confidence": "high",
                "source": "algorithm",
            })

        elif pair.alignment_type in ("added", "removed") and pair.v2_clause is None:
            change_counter += 1
            c1 = pair.v1_clause
            all_changes.append({
                "id": f"diff-{change_counter:03d}",
                "change_type": "removed",
                "clause_ref_v1": f"{c1.number} {c1.title}" if c1 else None,
                "clause_ref_v2": None,
                "brief": f"删除条款: {c1.title if c1 else ''}",
                "v1_snippet": c1.full_text[:200] if c1 else None,
                "v2_snippet": None,
                "confidence": "high",
                "source": "algorithm",
            })

    return all_changes


def _diff_pair(
    client: OpenAI,
    pair: AlignedPair,
    model: str,
    max_tokens: int,
) -> dict | None:
    """Diff a single aligned clause pair via LLM."""
    prompt_info = _build_pair_prompt(pair)
    if prompt_info is None:
        return None

    title, user_prompt = prompt_info

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": IDENTIFIER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            timeout=300,
            stream=True,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        print(f"  ⚠️  LLM 调用失败 ({title[:30]}...): {e}")
        return None

    parts = []
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)

    content = "".join(parts)
    try:
        repaired = repair_json(content)
        raw = json.loads(repaired)
        return _normalize_changes(raw)
    except Exception as e:
        print(f"  ⚠️  JSON 解析失败 ({title[:30]}...): {e}")
        return None


def _normalize_changes(raw: dict) -> dict:
    """Normalize LLM output to our standard change schema."""
    if raw.get("no_change"):
        return {"no_change": True, "changes": []}

    changes = raw.get("changes", [])
    if not changes:
        return {"no_change": True, "changes": []}

    normalized = []
    for i, c in enumerate(changes):
        nc = {
            "id": str(c.get("id", i + 1)),
            "change_type": _infer_change_type(c),
            "brief": c.get("brief") or c.get("description") or c.get("summary", ""),
            "v1_snippet": c.get("v1_snippet") or c.get("v1", ""),
            "v2_snippet": c.get("v2_snippet") or c.get("v2", ""),
            "confidence": c.get("confidence", "medium"),
        }
        # Remove None values and empty strings
        for key in list(nc.keys()):
            if nc[key] is None or nc[key] == "":
                del nc[key]
        normalized.append(nc)
    return {"changes": normalized, "no_change": False}


def _infer_change_type(c: dict) -> str:
    raw_type = c.get("change_type") or c.get("type") or c.get("category", "")
    if "新增" in raw_type or "add" in raw_type.lower():
        return "added"
    if "删除" in raw_type or "remov" in raw_type.lower():
        return "removed"
    if "修改" in raw_type or "变更" in raw_type or "modif" in raw_type.lower():
        return "modified"
    # Heuristic based on content
    if not c.get("v1") and c.get("v2"):
        return "added"
    if c.get("v1") and not c.get("v2"):
        return "removed"
    return "modified"
