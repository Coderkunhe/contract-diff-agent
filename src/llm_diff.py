"""LLM-powered semantic contract comparison.

Uses OpenAI-compatible API (GMI proxy) to call Claude models.
"""

import json
import re
from datetime import datetime, timezone

from json_repair import repair_json
from .model_pool import AutoFallbackClient

from .pdf_extractor import ContractDocument, estimate_tokens

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

SYSTEM_PROMPT = """你是一名风控合规分析师，专门比对合同版本差异。你的输出面向非法律背景的
业务人员，要求语言通俗、结构清晰。

## 你的任务

对比以下两份合同版本，找出所有实质性差异，并按风险类型分类。

## 风险分类标准

{risk_categories}

## 输出要求

1. **聚焦重点**：只列出风险等级为 high 和 medium 的变化，low 风险（格式/编号调整）可忽略
2. **限量输出**：最多列出 20-25 个最重要的变化，按重要性排序
3. **精炼表达**：brief 控制在30字以内，risk_note 控制在60字以内，snippet 控制在100字以内
4. **准确定位**：尽可能标注条款编号

## 重要约束

- 不要给出合规结论或法律建议
- 不要评价合同整体优劣
- 只描述变化和潜在影响

## JSON 格式规则（必须严格遵守）
- 所有字符串值中不得出现英文双引号 \"
- 如需引用原文，请使用「」或『』代替双引号
- 确保 output JSON 本身是合法可解析的

## 输出格式

直接输出一个 JSON 对象，不要使用 markdown 代码块。严格按照以下结构：
    "total_changes": <int>,
    "added": <int>,
    "removed": <int>,
    "modified": <int>,
    "coverage_pct": <float, 0.0-1.0>
  }},
  "changes": [
    {{
      "id": "diff-001",
      "change_type": "added|removed|modified",
      "clause_ref_v1": "条款编号或 null",
      "clause_ref_v2": "条款编号或 null",
      "clause_title": "变化标题",
      "brief": "一句话变化摘要（通俗语言）",
      "v1_snippet": "V1原文关键句或 null",
      "v2_snippet": "V2原文关键句或 null",
      "risk_categories": ["R01"],
      "risk_level": "high|medium|low",
      "risk_note": "通俗风险提示",
      "attention_for": "建议关注部门或 null",
      "is_favorable": true/false/null
    }}
  ],
  "risk_taxonomy_snapshot": {{
    "categories_used": ["R01"],
    "new_categories_discovered": [],
    "high_frequency_alerts": []
  }},
  "unmatched_content": {{
    "v1_only": [],
    "v2_only": [],
    "note": "说明"
  }}
}}"""

USER_PROMPT_TEMPLATE = """## 合同 V1（旧版）

{v1_text}

---

## 合同 V2（新版）

{v2_text}

---

请逐章节比对以上两份合同，找出所有差异点，对每个差异进行风险分类。
直接输出 JSON，不要包含任何解释文字。"""


def _build_system_prompt() -> str:
    cat_lines = "\n".join(
        f"- {c['id']}: {c['name']}（关注：{c['focus']}）"
        for c in RISK_CATEGORIES
    )
    return SYSTEM_PROMPT.format(risk_categories=cat_lines)


def _build_user_prompt(v1_text: str, v2_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(v1_text=v1_text, v2_text=v2_text)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling truncation and malformed JSON."""
    candidate = ""

    # Handle ```json ... ``` block (may be truncated without closing ```)
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*)", text)
    if m:
        candidate = m.group(1).strip()
        # Strip trailing ``` if present
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]

    if not candidate:
        raise ValueError(f"Cannot extract JSON from response:\n{text[:500]}")

    # Try parsing as-is, then apply fixes
    for fix_name, fix_fn in [
        ("raw", lambda s: json.loads(s)),
        ("json_repair", lambda s: json.loads(repair_json(s))),
        ("truncated_repair", _parse_truncated_repair),
        ("brute_force", _parse_brute_force),
    ]:
        try:
            return fix_fn(candidate)
        except (json.JSONDecodeError, Exception) as e:
            if fix_name == "raw":
                print(f"  ⚠️  JSON 解析失败, 自动修复中...")
            continue

    raise ValueError(f"Cannot parse JSON. Raw (first 500 chars):\n{candidate[:500]}")


def _parse_truncated_repair(s: str) -> dict:
    """Try json_repair on truncated content."""
    repaired = repair_json(s)
    result = json.loads(repaired)
    if result.get("changes"):
        print(f"  ✅ JSON 已修复, {len(result['changes'])} 条变化")
    return result


def _parse_truncated(s: str) -> dict:
    """Salvage truncated JSON by finding the last complete change object."""
    last_complete = 0
    for m in re.finditer(r'\},\s*\{', s):
        last_complete = m.end() - 1
    if last_complete > 0:
        # Close changes array, add empty remaining fields, close outer object
        salvaged = (
            s[:last_complete]
            + "],"
            + '"risk_taxonomy_snapshot": {"categories_used": [], "new_categories_discovered": [], "high_frequency_alerts": []},'
            + '"unmatched_content": {"v1_only": [], "v2_only": [], "note": "响应被截断, 部分内容可能缺失"}'
            + "}"
        )
        result = json.loads(salvaged)
        if result.get("changes"):
            print(f"  ⚠️  响应截断, 已恢复 {len(result['changes'])} 条变化")
        return result
    raise json.JSONDecodeError("No complete changes found", s, 0)


def _parse_brute_force(s: str) -> dict:
    """Try every truncation point from end, looking for valid JSON."""
    # Work backwards in chunks for efficiency
    for i in range(len(s) - 1, max(len(s) - 2000, 0), -50):
        try:
            return json.loads(s[:i])
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("Brute force failed", s, 0)


def run_llm_diff(
    v1: ContractDocument,
    v2: ContractDocument,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4.6",
    max_tokens: int = 3000,
    base_url: str = "https://api.gmi-serving.com/v1",
) -> dict:
    """Run semantic contract comparison using GMI proxy -> Claude."""

    client = AutoFallbackClient(api_key=api_key, base_url=base_url,
                                primary_model=model, timeout=300.0)

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(v1.full_text, v2.full_text)

    input_tokens = estimate_tokens(v1.full_text) + estimate_tokens(v2.full_text)
    print(f"  模型池: {model} (自动切换)")
    print(f"  输入 tokens 估算: ~{input_tokens:,}")

    response = client.create(
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        response_format={"type": "json_object"},
    )

    # Collect streamed chunks
    content_parts: list[str] = []
    usage_info = None
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content_parts.append(chunk.choices[0].delta.content)
        if hasattr(chunk, "usage") and chunk.usage:
            usage_info = chunk.usage

    content = "".join(content_parts)
    tool_output = _extract_json(content)

    if usage_info:
        print(f"  实际用量: prompt={usage_info.prompt_tokens}, "
              f"completion={usage_info.completion_tokens}")
    else:
        print(f"  响应长度: {len(content)} chars, tokens={input_tokens}")

    actual_tokens = {
        "prompt": usage_info.prompt_tokens if usage_info else input_tokens,
        "completion": usage_info.completion_tokens if usage_info else 0,
    }

    # Normalize field names (model sometimes uses "summary" instead of "diff_summary")
    if "summary" in tool_output and "diff_summary" not in tool_output:
        tool_output["diff_summary"] = tool_output.pop("summary")

    result = {
        "meta": {
            "contract_v1": v1.file_path,
            "contract_v2": v2.file_path,
            "compared_at": datetime.now(timezone.utc).isoformat(),
            "agent_version": "0.2.0",
            "model": model,
            "token_estimate": {
                "v1_tokens": estimate_tokens(v1.full_text),
                "v2_tokens": estimate_tokens(v2.full_text),
                "total": input_tokens,
            },
            "actual_tokens": actual_tokens,
        },
        **tool_output,
    }

    return result
