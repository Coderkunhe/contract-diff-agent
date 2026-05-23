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

### 1. 创建 .env

```bash
cp .env.example .env
```

### 2. 配置 API Provider（至少配一个）

项目支持两种 API 接入方式，**配一个就能跑**，两个都配则互为备份：

#### 方式 A：DeepSeek 直连（推荐，有免费额度）

| 步骤 | 说明 |
|------|------|
| 注册 | https://platform.deepseek.com |
| 获取 Key | 控制台 → API Keys → 新建 |
| 充值 | 新用户赠送免费额度，用完后按量付费（¥1/百万 tokens） |
| 模型 | `deepseek-chat`（V3 快）, `deepseek-reasoner`（R1 推理强） |

```bash
# .env 配置
DEEPSEEK_API_KEY=sk-xxxxxxxx
CLAUDE_MODEL=deepseek-chat
```

> **注意**：DeepSeek 不支持图片识别，但本项目是纯文本合同比对，不受影响。

#### 方式 B：GMI 多模型代理

| 步骤 | 说明 |
|------|------|
| 获取 Key | 联系 GMI 服务方获取 JWT Token |
| 充值 | 在 GMI 平台充值，余额不足时所有模型均不可用 |
| 模型 | Claude、GPT、Qwen、GLM 等 8+ 模型 |

```bash
# .env 配置
ANTHROPIC_API_KEY=你的GMI-JWT-Token
GMI_BASE_URL=https://api.gmi-serving.com/v1
CLAUDE_MODEL=anthropic/claude-sonnet-4.6
```

### 3. 模型自动切换

系统内置模型池，当前模型不可用时**自动降级切换**：

| 优先级 | 模型 | Provider | 说明 |
|--------|------|----------|------|
| 1 | deepseek-chat | DeepSeek | 快、中文好、有免费额度 |
| 2 | deepseek-reasoner | DeepSeek | 推理能力强，复杂合同用 |
| 3 | claude-sonnet-4.6 | GMI | 结构化输出最优 |
| 4 | claude-opus-4.7 | GMI | 最强综合能力 |
| 5 | Qwen/Qwen3.7-Max | GMI | 中文最优 |
| 6-10 | GPT、GLM 等 | GMI | 兜底 |

切换逻辑：
- 当前模型失败 → 自动尝试下一个 → 全失败则报错
- 失败模型进入冷却期（30s 起步，指数增长到 5 分钟）
- 成功一次自动重置冷却

> **省钱小贴士**：只配 DeepSeek 就够用。两把钥匙都配是为了"东边不亮西边亮"——一个平台挂了自动切另一个。

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
