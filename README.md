# ContractLens — 合同差异智能比对

风控合规 Agent，辅助识别合同不同版本之间的差异，将潜在风险以通俗、结构化的方式提示给非专业人员。

## 快速开始

```bash
make install      # 创建 venv + 安装依赖
make web          # 启动 Web 服务 → http://localhost:8000
make test         # 运行 106 个单元测试
make run-offline  # CLI 离线模式（无需 API Key）
make run          # CLI 完整模式（需 API Key）
```

## 架构 (v1.5.0)

```
                         ┌──────────────────────────────────────────┐
  PDF / DOCX            │          ContractLens Pipeline            │
  ─────────             │                                          │
     │                  │  ┌──────────┐   ┌──────────┐             │
     ▼                  │  │ ① 条款树  │──▶│ ② 条款对齐│             │
  ┌─────────┐           │  │  解析     │   │ 章节匹配  │             │
  │ pdfplumber│          │  │ L1/L2/L3  │   │ 序偏移处理│             │
  │ python-docx│         │  └──────────┘   └──────────┘             │
  └─────────┘           │        │              │                  │
     │                  │        ▼              ▼                  │
     ▼                  │  ┌─────────────────────────┐             │
  ┌─────────┐           │  │ ③ 传统算法 Diff (离线可用) │             │
  │ 文本提取 │           │  │ SequenceMatcher         │             │
  │ 双语过滤 │           │  │ 无关Diff过滤 · 分段合并  │             │
  └─────────┘           │  └─────────────────────────┘             │
                         │        │              │                  │
                         │        ▼              ▼                  │
                         │  ┌──────────┐   ┌──────────────┐        │
                         │  │ 基础报告  │   │ ④ LLM 增强   │        │
                         │  │ (离线)    │   │ (可选, 并行) │        │
                         │  └──────────┘   │ Claude/GPT/  │        │
                         │                  │ DeepSeek/Qwen│        │
                         │                  └──────────────┘        │
                         │                        │                 │
                         │                        ▼                 │
                         │               ┌─────────────────┐       │
                         │               │ ⑤ L2/L3 校验    │       │
                         │               │ L2 字符串原文验证 │       │
                         │               │ L3 LLM 交叉校验  │       │
                         │               └─────────────────┘       │
                         │                        │                 │
                         │                        ▼                 │
                         │               ┌─────────────────┐       │
                         │               │ ⑥ 风险分类       │       │
                         │               │ R01-R10 风险分类  │       │
                         │               │ 严谨模式: 独立    │       │
                         │               │ 交叉验证         │       │
                         │               └─────────────────┘       │
                         └──────────────────────────────────────────┘
                                             │
                                    ┌────────┴────────┐
                                    ▼                 ▼
                              ┌──────────┐     ┌──────────┐
                              │ JSON 报告 │     │ PDF 报告  │
                              │ + Web UI  │     │ (含备注)  │
                              └──────────┘     └──────────┘
                                    │
                                    ▼
                              ┌──────────────────┐
                              │ ⑦ 自进化学习      │
                              │ 经验提取 → 注入   │
                              │ 下次 LLM 提示词   │
                              └──────────────────┘
```

### 管线步骤

| 步骤 | 技术 | 说明 |
|------|------|------|
| ① 条款树解析 | pdfplumber + 正则 | 提取章节层级 (L1/L2/L3)、表格、附件；过滤双语噪音 |
| ② 条款对齐 | 标题相似度匹配 | 处理章节编号偏移、结构重组、中英文标题混排 |
| ③ 传统 Diff | SequenceMatcher | **零 LLM 依赖**，离线可用；无关差异过滤（纯空格/标点）；相邻块智能合并 |
| ④ LLM 增强 | Claude/GPT/DeepSeek/Qwen 并行 | 自然语言润色 + 风险分类；失败自动降级不丢数据 |
| ⑤ 校验 | L2 字符串匹配 + L3 LLM 交叉验证 | 反幻觉：L2 逐字确认摘要中的原文关键词 → L3 独立 LLM 复核 |
| ⑥ 风险分类 | R01-R10 分类体系 | 10 大风险类别；严谨模式独立交叉验证 |
| ⑦ 自进化 | 经验提取 → 提示词注入 | 每轮比对自动学习风险模式，下次 LLM 调用自动注入历史经验 |

**核心设计**: 传统算法是基础层，LLM 是可选增强层。断网、API 故障时系统仍产出完整的人眼可读报告。

---

## 技术创新点

### 1. 传统算法优先架构（Offline-First）

LLM 对合同比对任务中常见的"长文本逐句对比"有天然的上下文窗口限制和幻觉风险。本系统将 Diff 计算放在传统算法层（SequenceMatcher），LLM 只做自然语言增强和风险解读——**算法干重活，LLM 做翻译**。离线模式下零 API 依赖仍可产出 393 条结构化差异。

### 2. 多模型池自动切换

10+ 模型覆盖 DeepSeek、Claude、GPT、Qwen 四条产品线。模型失败时指数退避冷却、自动降级切换到备用模型。Provider 根据模型 ID 前缀自动识别（deepseek-* → DeepSeek API，其他 → GMI 代理），杜绝单点故障。

### 3. L2+L3 双重校验反幻觉

- **L2**: 将 LLM 输出的摘要关键词逐一在原文中做模糊字符串匹配
- **L3**: 用不同 LLM 模型独立复核原始 LLM 的判断，不一致则标记 `uncertain` 或 `rejected`

两重校验后置信度低于 0.6 的差异自动标记，防止幻觉进入最终报告。

### 4. 自进化学习闭环

每轮比对完成后自动提取：风险模式分布、LLM 校验拒绝率、高置信度模式、人工纠正记录。下次 LLM 调用时自动注入历史经验上下文（"过往 5 次比对中，交付时效（R01）频繁出现……请优先检查"），形成 **运行越多越准确** 的正循环。

### 5. SSE 实时流式进度

Web 端通过 Server-Sent Events 实时推送管线进度、逐条差异变化，非轮询方案。前端 Alpine.js 组件响应式渲染，SVG 环形进度条分阶段动画。

### 6. 严谨模式（Thorough Mode）

开启后，风险分类阶段使用**独立于增强阶段的 LLM 调用**进行交叉验证——不共享上下文、不共享模型状态，本质上是一次独立的"盲审"。准确率更高，耗时约 2 倍。

---

## 离线模式

```bash
# 无需任何 API Key，纯算法对比
python -m src.main v1.pdf v2.pdf --offline -o result.json
```

---

## 配置

### 1. 创建 .env

```bash
cp .env.example .env
```

### 2. 配置 API（可选，离线模式不需要）

**最小配置只需两行**：

```bash
LLM_API_KEY=sk-your-key-here
LLM_MODEL=deepseek-chat
```

`LLM_MODEL` 决定 Provider（自动识别）：
- `deepseek-*` → 直连 DeepSeek API (`https://api.deepseek.com/v1`)
- 其他模型 ID → GMI 代理 (`https://api.gmi-serving.com/v1`)

> ⚠️ **单模型提醒**：本项目设计为多模型池架构。仅配置单一模型意味着模型不可用时系统无法自动降级切换。
> 如需多模型池，可为不同 Provider 设置独立的 Key（详见 `.env.example` 注释）。

### 3. 可调参数

所有参数均设有合理默认值，可按需在 `.env` 中覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MAX_TOKENS` | 3000 | LLM 单次调用最大输出 token |
| `LLM_TIMEOUT` | 300 | HTTP 请求超时（秒） |
| `LLM_BATCH_SIZE` | 25 | 风险分类批量大小 |
| `CLASSIFY_MAX_TOKENS` | 2500 | 分类阶段 max_tokens |
| `ENHANCE_MAX_TOKENS` | 2000 | 增强描述阶段 max_tokens |
| `VALIDATE_MAX_TOKENS` | 500 | 校验阶段 max_tokens |
| `MAX_RETRIES` | 3 | 单条 LLM 调用最大重试 |
| `CHAPTER_RETRY_LIMIT` | 5 | 单章总循环上限 |
| `CONFIDENCE_THRESHOLD` | 0.6 | 置信度阈值（低于此值标记 uncertain） |
| `MODEL_COOLDOWN_BASE` | 30 | 模型失败冷却基数（秒，翻倍递增） |
| `MODEL_COOLDOWN_MAX` | 300 | 模型冷却上限（秒） |
| `ENHANCE_WORKERS` | 5 | 增强阶段并行度 |
| `VALIDATE_WORKERS` | 8 | 校验阶段并行度 |
| `MAX_JOBS` | 20 | Web 服务最大并发作业数 |
| `DATA_DIR` | data | 数据存储目录 |

---

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

---

## Web UI

| 页面 | 功能 |
|------|------|
| `/` | 上传页：拖拽上传 PDF/DOCX、文件大小显示、严谨模式开关 |
| `/job/{id}` | 进度页：SSE 实时流式、SVG 环形进度条、逐条差异即时展示 |
| `/results/{id}` | 结果页：风险筛选、类型筛选、备注自动保存（失焦即存）、右侧悬浮确认导航、PDF 导出 |
| `/learnings` | 进化记录：累计运行统计、风险趋势、排序/筛选 |

技术栈：**FastAPI + Jinja2 + Tailwind CSS + Alpine.js**，零前端构建。

---

## 测试

```bash
make test   # 106 个测试，覆盖：提取/解析/对齐/传统Diff/L2校验/自进化/Web路由
```

```
tests/test_clause_aligner.py    — 条款对齐 (10)
tests/test_clause_tree.py       — 条款树解析 (8)
tests/test_pdf_extractor.py     — PDF 提取 + Token估算 (9)
tests/test_traditional_diff.py  — 传统 Diff 算法 (19)
tests/test_validator.py         — L2/L3 校验 (7)
tests/test_learning.py          — 自进化学习 (21)
tests/test_web_routes.py        — Web 路由集成 (28)
```

---

## 项目结构

```
src/
  main.py                 # CLI 入口
  config.py               # 配置管理（单例, 18 个可调参数）
  pipeline/               # 流水线模块
    extraction.py          # ① PDF/DOCX 提取
    parsing.py             # ① 条款树解析 (L1/L2/L3)
    alignment.py           # ② 条款对齐 (标题相似度)
    traditional_diff.py    # ③ 传统算法对比 (零 LLM 依赖)
    identifier.py          # ④ LLM 增强描述
    validator.py           # ⑤ L2/L3 校验
    classifier.py          # ⑥ 风险分类
    learning.py            # ⑦ 自进化学习提取器
    diff.py                # 全文本 LLM Diff (v0.2 兼容)
  prompts/                 # LLM 提示语 (独立管理)
  constants/
    risks.py               # R01-R10 风险分类种子
  llm/                     # LLM 调用层
    client.py              # AutoFallbackClient
    pool.py                # 10+ 模型池配置
  utils/
    logging.py             # 日志配置
web/
  app.py                   # FastAPI + SSE 流式推送
  templates/               # Jinja2 模板 (Tailwind + Alpine.js)
    base.html.jinja2
    upload.html.jinja2     # 拖拽上传
    job.html.jinja2        # 实时进度
    results.html.jinja2    # 结果 + 悬浮导航
    learnings.html.jinja2  # 自进化记录
tests/                     # pytest 106 单测
specs/                     # 需求文档 / 架构决策记录
```

---

## 许可

供内部审核使用，不构成正式法律意见。
