# ContractLens — 合同差异智能比对

风控合规 Agent，辅助识别合同不同版本之间的差异，将潜在风险以通俗、结构化的方式提示给非专业人员。

## 快速开始

```bash
make install    # 创建 venv + 安装依赖
make web        # 启动 Web 服务 → http://localhost:8000
make test       # 运行 36 个单元测试
make run        # CLI 模式直接比对
```

## 架构

```
PDF → ①条款树解析(算法) → ②条款对齐(算法) → ③差异识别(LLM+并行)
    → ④校验(L2算法+L3 LLM) → ⑤风险分类(LLM) → JSON/Markdown输出 → Web UI
```

| 步骤 | 技术 | 说明 |
|------|------|------|
| ① 条款树 | pdfplumber + 正则 | 提取章节层级、表格、附件 |
| ② 对齐 | 标题相似度匹配 | 处理章节编号偏移、结构重组 |
| ③ 识别 | Claude/GPT/Qwen (5路并行) | 语义级差异比对 + 风险打标 |
| ④ 校验 | 字符串匹配 + LLM 交叉验证 | L2 原文验证 + L3 反幻觉 |
| ⑤ 分类 | LLM 批量 | 10 类风险种子 + 动态增长 + 频次统计 |

## 配置

复制 `.env.example` 为 `.env`，填入 API Key：

```bash
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY
```

支持的模型（通过 GMI 代理，自动切换）：

| 优先级 | 模型 | 定位 |
|--------|------|------|
| 1 | anthropic/claude-sonnet-4.6 | 主力 |
| 2 | anthropic/claude-opus-4.7 | 最强 |
| 3 | Qwen/Qwen3.7-Max | 中文最优 |
| 4 | deepseek-ai/DeepSeek-V4-Pro | 推理强 |
| ... | 共 11 个模型 | 自动降级切换 |

## 输出示例

```json
{
  "id": "diff-001",
  "change_type": "modified",
  "brief": "验收期限从7个工作日缩短为3个工作日",
  "risk_categories": ["R01"],
  "risk_level": "high",
  "risk_note": "验收时间缩短可能导致来不及充分检查",
  "attention_for": "采购部、法务部"
}
```

## 项目结构

```
src/            # 核心管道
  main.py       # CLI 入口
  pdf_extractor.py   # ① PDF 提取
  clause_tree.py     # ① 条款树
  clause_aligner.py  # ② 条款对齐
  diff_identifier.py # ③ LLM 差异识别
  validator.py       # ④ 校验
  risk_classifier.py # ⑤ 风险分类
  model_pool.py      # 动态模型池
web/            # Web UI
  app.py        # FastAPI + SSE 流式
  templates/    # Jinja2 模板 (Tailwind + Alpine.js)
tests/          # pytest 36 单测
```

## 许可

供内部审核使用，不构成正式法律意见。
