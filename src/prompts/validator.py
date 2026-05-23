"""Validation prompts.

Edit VALIDATOR_SYSTEM and VALIDATOR_USER to tune validation behavior.
"""

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


def build_validator_prompt(v1_snippet: str, v2_snippet: str, claim: str) -> str:
    return VALIDATOR_USER.format(
        v1_text=v1_snippet[:1000] or "（无）",
        v2_text=v2_snippet[:1000] or "（无）",
        claim=claim,
    )
