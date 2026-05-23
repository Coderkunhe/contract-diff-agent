"""Risk category taxonomy — single source of truth.

Used by prompt builders and classifier. Edit here to add/remove categories.
"""

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
