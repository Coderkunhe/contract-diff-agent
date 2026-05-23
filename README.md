# ContractLens — 合同差异智能比对

风控合规 Agent，辅助识别合同不同版本之间的差异，将潜在风险以通俗、结构化的方式提示给非专业人员。

## 快速开始

```bash
make install      # 创建 venv + 安装依赖
make web          # 启动 Web 服务 → http://localhost:8000
make test         # 运行 60 个单元测试
make run-offline  # CLI 离线模式（无需 API Key）
make run          # CLI 完整模式（需 API Key）
```

## 架构 (v1.4.0)

```
PDF → ①条款树解析 → ②条款对齐 → ③传统算法Diff（离线可用）
                                   ├── 基础报告（人眼可读）
                                   └── ④LLM增强（可选）→ ⑤L2/L3校验 → JSON/Web UI
```

| 步骤 | 技术 | 说明 |
|------|------|------|
| ① 条款树 | pdfplumber + 正则 | 提取章节层级、表格、附件 |
| ② 对齐 | 标题相似度匹配 | 处理章节编号偏移、结构重组 |
| ③ 传统 Diff | SequenceMatcher | 零 LLM 依赖，离线可用，产出结构化差异 |
| ④ LLM 增强 | Claude/GPT/Qwen (并行) | 自然语言润色 + 风险分类，失败不丢数据 |
| ⑤ 校验 | 字符串匹配 ± LLM 交叉验证 | L2 原文验证 + L3 反幻觉 |

**核心设计**：传统算法是基础层，LLM 是可选增强层。断网、API 故障时系统仍产出完整的人眼可读报告。

## 离线模式

```bash
# 无需任何 API Key，纯算法对比
python -m src.main v1.pdf v2.pdf --offline -o result.json
```

## 配置

### 1. 创建 .env

```bash
cp .env.example .env
```

### 2. 配置 API Provider（可选，离线模式不需要）

项目支持两种 API 接入方式，**配一个就能跑**，两个都配则互为备份：

#### 方式 A：DeepSeek 直连（推荐，有免费额度）

| 步骤 | 说明 |
|------|------|
| 注册 | https://platform.deepseek.com |
| 获取 Key | 控制台 → API Keys → 新建 |
| 充值 | 新用户赠送免费额度，用完后按量付费 |
| 模型 | `deepseek-chat`（V3 快）, `deepseek-reasoner`（R1 推理强） |

```bash
DEEPSEEK_API_KEY=sk-xxxxxxxx
CLAUDE_MODEL=deepseek-chat
```

#### 方式 B：GMI 多模型代理

```bash
ANTHROPIC_API_KEY=你的GMI-JWT-Token
GMI_BASE_URL=https://api.gmi-serving.com/v1
CLAUDE_MODEL=anthropic/claude-sonnet-4.6
```

### 3. 模型自动切换

当前模型不可用时自动降级切换：

| 优先级 | 模型 | Provider | 说明 |
|--------|------|----------|------|
| 1 | deepseek-chat | DeepSeek | 快、中文好、有免费额度 |
| 2 | deepseek-reasoner | DeepSeek | 推理能力强 |
| 3 | claude-sonnet-4.6 | GMI | 结构化输出最优 |
| 4 | claude-opus-4.7 | GMI | 最强综合能力 |
| 5 | Qwen/Qwen3.7-Max | GMI | 中文最优 |
| 6-10 | GPT、GLM 等 | GMI | 兜底 |

## 输出示例

```json
{
  "id": "diff-001",
  "change_type": "modified",
  "brief": "验收期限从7个工作日缩短为3个工作日",
  "risk_categories": ["R01"],
  "risk_level": "high",
  "risk_note": "验收时间缩短可能导致来不及充分检查",
  "attention_for": "采购部、法务部",
  "human_note": "已与采购部确认，新时限可接受"
}
```

## 项目结构

```
src/
  main.py               # CLI 入口
  config.py              # 配置管理
  pipeline/              # 流水线模块
    extraction.py        # ① PDF/DOCX 提取
    parsing.py           # ① 条款树解析
    alignment.py         # ② 条款对齐
    traditional_diff.py  # ③ 传统算法对比（零 LLM 依赖）
    identifier.py        # ④ LLM 增强描述
    validator.py         # ⑤ L2/L3 校验
    classifier.py        # 风险分类
    diff.py              # 全文本 LLM Diff（v0.2 兼容）
  prompts/               # LLM 提示语（独立管理）
    identifier.py
    classifier.py
    validator.py
    diff.py
  constants/             # 全局常量
    risks.py              # 风险分类种子
  llm/                   # LLM 调用层
    client.py             # AutoFallbackClient
    pool.py               # 模型池配置
  utils/
    logging.py            # 日志配置
web/                    # Web UI
  app.py                # FastAPI + SSE 流式
  templates/            # Jinja2 模板 (Tailwind + Alpine.js)
tests/                  # pytest 60 单测
specs/                  # 需求文档 / 架构决策记录
```

## 许可

供内部审核使用，不构成正式法律意见。
