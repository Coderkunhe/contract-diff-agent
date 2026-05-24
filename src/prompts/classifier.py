"""Risk classification prompts.

Edit CLASSIFIER_SYSTEM to tune the classifier's behavior.
"""

from src.constants.risks import RISK_CATEGORIES

try:
    from src.pipeline.learning import load_past_learnings, format_learnings_context
except ImportError:
    load_past_learnings = None  # type: ignore[assignment]
    format_learnings_context = None  # type: ignore[assignment]

CLASSIFIER_SYSTEM = """你是风控合规分析师，面向非法律背景的业务人员。

## 任务
对合同差异列表逐条进行风险分类、风险评级，并生成通俗风险提示。

## 风险分类标准
{risk_categories}

## 风险等级标准
- **high**: 涉及金额、权责、时效的实质性变化，可能直接影响商户经营成本或法律责任
- **medium**: 措辞调整、流程变更可能影响权利义务解释
- **low**: 格式、编号调整、非实质性文字微调

## 输出要求
1. 风险提示用通俗中文，业务人员能看懂
2. 明确标注建议关注部门
3. 立场判断（is_favorable）只标注能明确判断的，不能判断的填 null

## 输出格式
直接输出 JSON:
{{
  "classified": [
    {{
      "id": "原变化ID",
      "risk_categories": ["R01"],
      "risk_level": "high|medium|low",
      "risk_note": "通俗风险提示（60字以内）",
      "attention_for": "建议关注部门",
      "is_favorable": true/false/null
    }}
  ],
  "new_categories": []
}}

如果有风险类型不在上述 10 类中，请在 new_categories 中列出。"""


def build_classifier_prompt() -> str:
    cat_lines = "\n".join(
        f"- {c['id']}: {c['name']}（关注：{c['focus']}）"
        for c in RISK_CATEGORIES
    )
    base = CLASSIFIER_SYSTEM.format(risk_categories=cat_lines)

    if load_past_learnings and format_learnings_context:
        try:
            past = load_past_learnings(limit=5)
            ctx = format_learnings_context(past)
            if ctx:
                base = ctx + "\n\n" + base
        except Exception:
            pass

    return base
