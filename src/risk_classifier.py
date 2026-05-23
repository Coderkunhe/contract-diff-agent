"""Risk classification & friendly output - Step ⑤ of the contract diff pipeline.

Batches changes and calls LLM to assign risk categories, levels, and plain-language notes.
"""

import json
import os
from typing import Any

from json_repair import repair_json
from .model_pool import AutoFallbackClient
from .pdf_extractor import estimate_tokens

RISK_CATEGORIES = [
    {"id": "R01", "name": "交付时效", "focus": "交货期限、验收周期、逾期条款变化"},
    {"id": "R02", "name": "质量与鉴定标准", "focus": "成色/净度/认证标准变化"},
    {"id": "R03", "name": "付款与结算", "focus": "付款节点、账期、保证金、扣点变化"},
    {"id": "R04", "name": "违约责任", "focus": "违约金比例、赔偿上限、免责范围变化"},
    {"id": "R05", "name": "知识产权与品牌", "focus": "商标授权、设计版权、品牌使用规则变化"},
    {"id": "R06", "name": "退换货与售后", "focus": "退货条件、换货期限、维修责任变化"},
    {"id": "R07", "name": "保险与风险承担", "focus": "运输/仓储保险、在途风险承担方变化"},
    {"id": "R08", "name": "运输与物流", "focus": "承运资质、配送时效、签收规则变化"},
    {"id": "R09", "name": "价格调整机制", "focus": "调价条款、原料联动定价、强制促销条款"},
    {"id": "R10", "name": "数据与隐私", "focus": "客户数据使用、数据安全责任变化"},
]

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


def classify_changes(
    changes: list[dict],
    api_key: str,
    model: str = "anthropic/claude-sonnet-4.6",
    base_url: str = "https://api.gmi-serving.com/v1",
    batch_size: int = 25,
) -> list[dict]:
    """Batch classify changes using LLM.

    Groups changes into batches to minimize API calls while staying
    within the 3000 max_tokens limit.
    """
    if not changes:
        return []

    client = AutoFallbackClient(api_key=api_key, base_url=base_url,
                                primary_model=model, timeout=300.0)

    cat_lines = "\n".join(
        f"- {c['id']}: {c['name']}（关注：{c['focus']}）"
        for c in RISK_CATEGORIES
    )
    system_prompt = CLASSIFIER_SYSTEM.format(risk_categories=cat_lines)

    frequency: dict[str, int] = {c["id"]: 0 for c in RISK_CATEGORIES}
    new_categories: list[dict] = []
    classified: list[dict] = []
    total_batches = (len(changes) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(changes), batch_size):
        batch = changes[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        print(f"  分类进度: batch {batch_num}/{total_batches} "
              f"({len(batch)} 条变化)")

        # Build compact input
        changes_json = []
        for c in batch:
            changes_json.append({
                "id": c.get("id", ""),
                "change_type": c.get("change_type", "modified"),
                "brief": c.get("brief", ""),
                "clause": c.get("clause_ref_v2") or c.get("clause_ref_v1", ""),
            })

        user_prompt = f"风险分类以下合同差异:\n{json.dumps(changes_json, ensure_ascii=False)}"

        try:
            response = client.create(
                max_tokens=2500,
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
            result = json.loads(repair_json(content))

            # Merge classifications back into changes
            lookup = {c["id"]: c for c in result.get("classified", [])}
            for change in batch:
                cls = lookup.get(change.get("id", ""), {})
                change["risk_categories"] = cls.get("risk_categories", [])
                change["risk_level"] = cls.get("risk_level", "medium")
                change["risk_note"] = cls.get("risk_note", "")
                change["attention_for"] = cls.get("attention_for")
                change["is_favorable"] = cls.get("is_favorable")
                classified.append(change)

                # Track frequency
                for cat_id in change.get("risk_categories", []):
                    frequency[cat_id] = frequency.get(cat_id, 0) + 1

            for nc in result.get("new_categories", []):
                if nc not in new_categories:
                    new_categories.append(nc)

        except Exception as e:
            print(f"  ⚠️  分类失败 (batch {batch_num}): {e}")
            # Keep changes unclassified rather than losing them
            for change in batch:
                change.setdefault("risk_categories", [])
                change.setdefault("risk_level", "medium")
                change.setdefault("risk_note", "")
                classified.append(change)

    # Summary
    print(f"\n  分类完成: {len(classified)} 条")
    if frequency:
        top = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        cat_names = {c["id"]: c["name"] for c in RISK_CATEGORIES}
        print(f"  高频风险: {', '.join(f'{cat_names.get(k, k)}({v}次)' for k, v in top if v > 0)}")
    if new_categories:
        print(f"  新发现分类: {len(new_categories)}")

    return classified


def load_taxonomy(path: str = "data/risk_taxonomy.json") -> dict:
    """Load persisted risk taxonomy with frequency data."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"categories": RISK_CATEGORIES, "frequency": {}, "new_categories": []}


def save_taxonomy(frequency: dict, new_categories: list[dict],
                  path: str = "data/risk_taxonomy.json"):
    """Persist risk taxonomy with updated frequencies."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "categories": RISK_CATEGORIES,
        "frequency": frequency,
        "new_categories": new_categories,
    }
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
