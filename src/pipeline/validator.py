"""Validation agent — Step ④ of the contract diff pipeline.

L2: Algorithmic snippet existence check
L3: LLM semantic validation
With retry loop and exit conditions.
"""

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from json_repair import repair_json

from src.llm.client import AutoFallbackClient
from src.prompts.validator import build_validator_prompt, VALIDATOR_SYSTEM

_VALIDATE_MAX_WORKERS = 8


def validate_changes(
    changes: list[dict],
    v1_full_text: str,
    v2_full_text: str,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4.6",
    base_url: str = "https://api.gmi-serving.com/v1",
    max_retries: int = 2,
    max_workers: int = _VALIDATE_MAX_WORKERS,
) -> list[dict]:
    print(f"  L2 原文校验 ({len(changes)} 条)...")
    for change in changes:
        l2_result = _l2_check(change, v1_full_text, v2_full_text)
        change["_l2"] = l2_result

    llm_changes = [(i, c) for i, c in enumerate(changes) if c.get("source") == "llm"]
    if not llm_changes:
        for c in changes:
            c["validation"] = {"status": "verified", "l3_verdict": "algorithm",
                               "confidence": 1.0}
        return changes

    print(f"  L3 语义校验 ({len(llm_changes)} 条, {max_workers} 路并行)...")
    client = AutoFallbackClient(primary_model=model, timeout=300.0)
    stats_lock = threading.Lock()
    stats = {"confirmed": 0, "rejected": 0, "uncertain": 0}
    completed = 0

    def _validate_one(idx: int, change: dict) -> tuple[int, dict]:
        result = _l3_validate(change, v1_full_text, v2_full_text,
                              client, max_retries)
        return idx, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_validate_one, i, c): i for i, c in llm_changes}
        for future in as_completed(futures):
            idx, result = future.result()
            with stats_lock:
                verdict = result.get("verdict", "uncertain")
                stats[verdict] = stats.get(verdict, 0) + 1
                completed += 1
                changes[idx]["_l3"] = result
                if completed % 50 == 0:
                    print(f"    校验进度: {completed}/{len(llm_changes)} "
                          f"(confirmed={stats['confirmed']}, rejected={stats['rejected']}, "
                          f"uncertain={stats['uncertain']})")

    for change in changes:
        l2 = change.pop("_l2", {})
        l3 = change.pop("_l3", {})
        change["validation"] = {**l2}
        change["validation"].update({
            "l3_verdict": l3.get("verdict", "skipped"),
            "l3_reason": l3.get("reason", ""),
            "l3_attempts": l3.get("attempts", 0),
            "status": _status_from(l3.get("verdict"), change.get("source")),
            "confidence": _conf_from(l3.get("verdict")),
        })

    print(f"  校验完成: {len(changes)} 条, "
          f"confirmed={stats['confirmed']}, rejected={stats['rejected']}, "
          f"uncertain={stats['uncertain']}")

    return changes


def _status_from(verdict: str, source: str) -> str:
    if source != "llm":
        return "verified"
    return {"confirmed": "verified", "rejected": "rejected"}.get(verdict, "uncertain")


def _conf_from(verdict: str) -> float:
    return {"confirmed": 0.95, "rejected": 0.1}.get(verdict, 0.5)


def _l2_check(change: dict, v1_text: str, v2_text: str) -> dict:
    result = {
        "l2_v1_snippet_found": False,
        "l2_v2_snippet_found": False,
        "l2_clause_ref_valid": False,
    }

    v1_snippet = change.get("v1_snippet")
    v2_snippet = change.get("v2_snippet")

    if v1_snippet:
        if v1_snippet in v1_text:
            result["l2_v1_snippet_found"] = True
        elif len(v1_snippet) > 30 and v1_snippet[:30] in v1_text:
            result["l2_v1_snippet_found"] = True
        else:
            words = re.findall(r"[一-鿿\w]+", v1_snippet)
            longest = max(words, key=len) if words else ""
            if longest and len(longest) > 4 and longest in v1_text:
                result["l2_v1_snippet_found"] = True
    else:
        result["l2_v1_snippet_found"] = True

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
    client: AutoFallbackClient,
    max_retries: int,
) -> dict:
    claim = change.get("brief", "")
    v1_snippet = change.get("v1_snippet") or ""
    v2_snippet = change.get("v2_snippet") or ""

    prompt = build_validator_prompt(v1_snippet, v2_snippet, claim)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.create(
                max_tokens=500,
                messages=[
                    {"role": "system", "content": VALIDATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
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
                prompt = build_validator_prompt(v1_snippet, v2_snippet, claim) + \
                         f"\n\n（上次验证失败: {e}。请重试。）"
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
