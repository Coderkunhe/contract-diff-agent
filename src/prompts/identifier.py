"""Clause-level diff prompts.

Two modes:
- IDENTIFIER_* — old: full clause diff from scratch (used when no traditional diff base)
- ENHANCE_* — new: take traditional diff output, enrich with natural language
"""

from src.constants.risks import RISK_CATEGORIES

# ── Enhancement mode (new) ────────────────────────────────────────
# Input: already-identified changes with snippets. LLM enriches.

ENHANCE_SYSTEM = """你是一名合同比对专家。下面是一份已经由算法识别出的合同差异列表。

你的任务：为每条差异用通俗语言重写摘要，并进行风险分类。

## 风险分类标准
{risk_categories}

## 风险等级
- high: 涉及金额、权责、时效的实质性变化，可能直接影响商户经营成本或法律责任
- medium: 措辞调整、流程变更可能影响权利义务解释
- low: 格式、编号、非实质性文字调整

## 输出 JSON
{{
  "enhanced": [
    {{
      "id": "原变化ID（保持不变）",
      "brief": "一句话摘要（30字以内，通俗白话）",
      "risk_categories": ["R01"],
      "risk_level": "high|medium|low",
      "risk_note": "通俗风险提示（50字以内）",
      "attention_for": "建议关注部门或null",
      "is_favorable": true/false/null
    }}
  ]
}}

is_favorable: 对我方（商户/甲方）有利填true, 不利填false, 无法判断填null
risk_note: 面向非法律人员，通俗易懂。
如果某条差异是算法误报（实际无变化），请把 risk_level 设为 low，risk_note 填「疑似误报」"""

ENHANCE_USER = """## 条款：{clause_title}
## V1 ({v1_ref}):
{v1_text}

## V2 ({v2_ref}):
{v2_text}

## 算法识别的差异：
{changes_json}

请逐条用通俗语言重写 brief，并补充风险分类。输出 JSON。"""


def build_enhance_system_prompt() -> str:
    cat_lines = "\n".join(
        f"- {c['id']}: {c['name']}（关注：{c['focus']}）"
        for c in RISK_CATEGORIES
    )
    return ENHANCE_SYSTEM.format(risk_categories=cat_lines)


def build_enhance_user_prompt(changes: list[dict], clause_title: str,
                               v1_ref: str, v2_ref: str,
                               v1_text: str, v2_text: str) -> str:
    compact = []
    for c in changes:
        compact.append({
            "id": c.get("id", ""),
            "change_type": c.get("change_type", "modified"),
            "v1_snippet": c.get("v1_snippet", ""),
            "v2_snippet": c.get("v2_snippet", ""),
            "brief": c.get("brief", ""),
        })
    import json
    return ENHANCE_USER.format(
        clause_title=clause_title,
        v1_ref=v1_ref, v2_ref=v2_ref,
        v1_text=v1_text[:2000], v2_text=v2_text[:2000],
        changes_json=json.dumps(compact, ensure_ascii=False),
    )


# ── Original identification mode (kept for reference) ──────────────

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


def build_system_prompt(skip_risk: bool = False) -> str:
    cat_lines = "\n".join(
        f"- {c['id']}: {c['name']}（关注：{c['focus']}）"
        for c in RISK_CATEGORIES
    )
    prompt = IDENTIFIER_SYSTEM.format(risk_categories=cat_lines)
    if skip_risk:
        prompt = prompt.replace(
            '"risk_categories": ["R01"],\n      "risk_level": "high|medium|low",\n      "risk_note": "通俗风险提示（50字以内）",\n      "attention_for": "建议关注部门或null",\n      "is_favorable": true/false/null',
            '"risk_categories": [], "risk_level": "low", "risk_note": "", "attention_for": null, "is_favorable": null'
        )
    return prompt


def build_pair_prompt(v1_clause, v2_clause, title: str,
                      v1_ref: str, v2_ref: str,
                      v1_text: str, v2_text: str) -> str:
    return IDENTIFIER_USER.format(
        clause_title=title,
        v1_ref=v1_ref,
        v2_ref=v2_ref,
        v1_text=v1_text[:3000],
        v2_text=v2_text[:3000],
    )
