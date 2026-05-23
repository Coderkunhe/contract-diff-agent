# JSON 输出结构定义

## 版本: v1.0-draft
## 说明: 所有模块输出的对齐基准，后续变更需同步更新本文件

## 完整结构

```json
{
  "meta": {
    "contract_v1": "string, V1合同标识（文件名/标题）",
    "contract_v2": "string, V2合同标识",
    "compared_at": "string, ISO8601 timestamp",
    "agent_version": "string, 当前Agent版本号"
  },
  "diff_summary": {
    "total_changes": "int",
    "added": "int",
    "removed": "int",
    "modified": "int",
    "coverage_pct": "float, 条款对齐覆盖率 0.0-1.0"
  },
  "changes": [
    {
      "id": "string, 唯一标识 diff-001",
      "change_type": "enum: added | removed | modified",
      "clause_ref_v1": "string | null, V1条款编号",
      "clause_ref_v2": "string | null, V2条款编号",
      "clause_title": "string, 条款标题/主题概括",
      "brief": "string, 一句话变化摘要（面向非法律人员）",
      "v1_snippet": "string | null, V1原文关键片段",
      "v2_snippet": "string | null, V2原文关键片段",
      "risk_categories": ["string, 风险分类ID列表"],
      "risk_level": "enum: high | medium | low",
      "risk_note": "string, 通俗风险提示",
      "attention_for": "string | null, 建议关注方（采购/法务/财务/运营）",
      "is_favorable": "boolean | null, 对我方是否有利（null=中性/待判断）"
    }
  ],
  "risk_taxonomy_snapshot": {
    "categories_used": ["本次比对命中的分类列表"],
    "new_categories_discovered": ["本次新发现的分类"],
    "high_frequency_alerts": ["命中次数超过阈值的分类"]
  },
  "unmatched_content": {
    "v1_only": ["string, 仅在V1中存在且无法对齐的段落摘要"],
    "v2_only": ["string, 仅在V2中存在且无法对齐的段落摘要"],
    "note": "string, 解释未对齐原因（格式变动过大/新增章节等）"
  }
}
```

## 字段约定
- 所有 `snippet` 字段保留原文，不改写
- `brief` 和 `risk_note` 用通俗中文，避免法律术语
- `risk_level` 判断标准：high=涉及金额/权责/时效实质性变化；medium=措辞调整可能影响解释；low=格式/编号变化
- `is_favorable` 允许 null，表示纯中性变化或无法判断立场
