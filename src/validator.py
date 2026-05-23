"""Validation agent - Step ④ of the contract diff pipeline.

L2: Algorithmic snippet existence check
L3: LLM semantic validation
With retry loop and exit conditions.
"""

import json
import re
from typing import Any

from openai import OpenAI
from json_repair import repair_json

VALIDATOR_SYSTEM = """你是合同比对验证专家。你的任务是验证一条声称的合同差异是否真实存在。

## 输入
- V1 原文片段
- V2 原文片段
- 声称的变化描述

## 输出
输出一个 JSON：
{
  "verdict": "confirmed|rejected|uncertain",
  "reason": "简短理由（30字以内）"
}

- confirmed: 原文明确支持该变化
- rejected: 原文不支持（可能是幻觉或过度解读）
- uncertain: 原文模糊，无法确认

只输出 JSON。"""

VALIDATOR_USER = """## V1 原文:
{v1_text}

## V2 原文:
{v2_text}

## 声称的变化:
{claim}

验证该变化是否真实。输出 JSON。"""


def validate_changes(
    changes: list[dict],
    v1_full_text: str,
    v2_full_text: str,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4.6",
    base_url: str = "https://api.gmi-serving.com/v1",
    max_retries: int = 3,
) -> list[dict]:
    """Validate all changes through L2 (algorithm) and L3 (LLM).

    Exit conditions per change:
    - Max 3 retries for L3 validation
    - Overall: mark uncertain if still failing
    """
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)

    validated: list[dict] = []
    stats = {"confirmed": 0, "rejected": 0, "uncertain": 0}

    for i, change in enumerate(changes):
        if i % 20 == 0 and i > 0:
            print(f"  校验进度: {i}/{len(changes)} "
                  f"(confirmed={stats['confirmed']}, rejected={stats['rejected']}, "
                  f"uncertain={stats['uncertain']})")

        vc = dict(change)  # Copy
        vc["validation"] = {"status": "unchecked"}

        # L2: snippet existence check
        l2_result = _l2_check(change, v1_full_text, v2_full_text)
        vc["validation"].update(l2_result)

        # L3: LLM semantic validation (for LLM-generated changes)
        if change.get("source") == "llm":
            l3_result = _l3_validate(change, v1_full_text, v2_full_text,
                                     client, model, max_retries)
            vc["validation"].update(l3_result)
            stats[l3_result.get("verdict", "uncertain")] += 1
        else:
            vc["validation"]["l3_verdict"] = "confirmed"
            vc["validation"]["status"] = "verified"
            vc["validation"]["confidence"] = 1.0
            stats["confirmed"] += 1

        validated.append(vc)

    print(f"  校验完成: {len(validated)} 条, "
          f"confirmed={stats['confirmed']}, rejected={stats['rejected']}, "
          f"uncertain={stats['uncertain']}")

    return validated


def _l2_check(change: dict, v1_text: str, v2_text: str) -> dict:
    """Algorithmic check: verify snippets exist in original texts."""
    result = {
        "l2_v1_snippet_found": False,
        "l2_v2_snippet_found": False,
        "l2_clause_ref_valid": False,
    }

    v1_snippet = change.get("v1_snippet")
    v2_snippet = change.get("v2_snippet")

    if v1_snippet:
        # Fuzzy find: try exact first, then first 30 chars
        if v1_snippet in v1_text:
            result["l2_v1_snippet_found"] = True
        elif len(v1_snippet) > 30 and v1_snippet[:30] in v1_text:
            result["l2_v1_snippet_found"] = True
        else:
            # Try finding key words (longest word in snippet)
            words = re.findall(r"[一-鿿\w]+", v1_snippet)
            longest = max(words, key=len) if words else ""
            if longest and len(longest) > 4 and longest in v1_text:
                result["l2_v1_snippet_found"] = True
    else:
        result["l2_v1_snippet_found"] = True  # No snippet to check

    if v2_snippet:
        if v2_snippet in v2_text:
            result["l2_v2_snippet_found"] = True
        elif len(v2_snippet) > 30 and v2_snippet[:30] in v2_text:
            result["l2_v2_snippet_found"] = True
        else:
            words = re.findall(r"[一-鿿\w]+", v2_snippet)
            longest = max(words, key=len) if words else ""
            if longest and len(longest) > 4 and longest in v2_text:
                result["l2_v2_snippet_found"] = True
    else:
        result["l2_v2_snippet_found"] = True

    return result


def _l3_validate(
    change: dict,
    v1_text: str,
    v2_text: str,
    client: OpenAI,
    model: str,
    max_retries: int,
) -> dict:
    """LLM semantic validation with retry loop."""
    claim = change.get("brief", "")
    v1_snippet = change.get("v1_snippet") or ""
    v2_snippet = change.get("v2_snippet") or ""

    prompt = VALIDATOR_USER.format(
        v1_text=v1_snippet[:1000] or "（无）",
        v2_text=v2_snippet[:1000] or "（无）",
        claim=claim,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": VALIDATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                timeout=120,
                stream=True,
                response_format={"type": "json_object"},
            )

            parts = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)

            content = "".join(parts)
            result = json.loads(repair_json(content))

            verdict = result.get("verdict", "uncertain")
            return {
                "l3_verdict": verdict,
                "l3_reason": result.get("reason", ""),
                "l3_attempts": attempt,
                "status": _verdict_to_status(verdict),
                "confidence": _verdict_to_confidence(verdict),
            }

        except Exception as e:
            if attempt < max_retries:
                # Retry with error context in prompt
                prompt = VALIDATOR_USER.format(
                    v1_text=v1_snippet[:1000] or "（无）",
                    v2_text=v2_snippet[:1000] or "（无）",
                    claim=claim,
                ) + f"\n\n（上次验证失败: {e}。请重试。）"
            else:
                return {
                    "l3_verdict": "uncertain",
                    "l3_reason": f"验证失败(重试{max_retries}次): {e}",
                    "l3_attempts": attempt,
                    "status": "uncertain",
                    "confidence": 0.5,
                }

    return {
        "l3_verdict": "uncertain",
        "l3_reason": "超出最大重试次数",
        "l3_attempts": max_retries,
        "status": "uncertain",
        "confidence": 0.5,
    }


def _verdict_to_status(verdict: str) -> str:
    if verdict == "confirmed":
        return "verified"
    elif verdict == "rejected":
        return "rejected"
    return "uncertain"


def _verdict_to_confidence(verdict: str) -> float:
    if verdict == "confirmed":
        return 0.95
    elif verdict == "rejected":
        return 0.1
    return 0.5
