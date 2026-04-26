# OpenEvals Evaluation Examples

这个目录整理了一组基于 `OpenEvals`、`LangSmith`、`LangGraph` 和通义千问模型的 LLM 应用评估示例。项目通过多个小脚本演示如何使用 LLM-as-Judge 评估回答正确性、简洁性、相关性、RAG 质量、安全性、幻觉、代码正确性以及 Agent 执行计划一致性。

## 项目定位

本项目适合用于学习和演示：

- OpenEvals 内置评估器的使用方式
- LLM-as-Judge 的基本评估模式
- RAG 场景中的 helpfulness、groundedness、retrieval relevance 评估
- 代码生成场景中的 correctness 评估
- Agent 场景中的 plan adherence 评估
- LangSmith 数据集、实验和评估流水线集成
- 基于 LangGraph 的混合型财富管理投顾 Agent 评估

## 目录结构

```text
.
├── 1-correctness.py
├── 1-hybrid_wealth_advisor_langgraph_langsmith.py
├── 2-conciseness.py
├── 2-langsmith_testing_evaluation.py
├── 3-answer_relevance.py
├── 4-rag_helpfulness.py
├── 5-rag_groundedness.py
├── 6-rag_retrieval_relevance.py
├── 7-toxicity.py
├── 8-hallucination.py
├── 9-code_correctness.py
├── 10-code_correctness_with_reference.py
├── 11-plan_adherence.py
├── 12-openevals_evaluators.py
├── requirements.txt
└── readme.md
```

## 示例说明

| 文件 | 评估主题 | 说明 |
| --- | --- | --- |
| `1-correctness.py` | Correctness | 判断模型输出是否符合参考答案。 |
| `2-conciseness.py` | Conciseness | 判断回答是否简洁、直接、无冗余。 |
| `3-answer_relevance.py` | Answer Relevance | 判断回答是否切题、是否回应用户问题。 |
| `4-rag_helpfulness.py` | RAG Helpfulness | 判断 RAG 回答是否对用户有帮助。 |
| `5-rag_groundedness.py` | RAG Groundedness | 判断回答是否被检索上下文支撑，是否存在脱离资料的内容。 |
| `6-rag_retrieval_relevance.py` | Retrieval Relevance | 判断检索到的上下文是否与用户问题相关。 |
| `7-toxicity.py` | Toxicity | 判断输出是否包含有害、攻击性或不安全内容。 |
| `8-hallucination.py` | Hallucination | 判断输出是否存在幻觉或无依据信息。 |
| `9-code_correctness.py` | Code Correctness | 判断代码输出是否满足题目要求。 |
| `10-code_correctness_with_reference.py` | Code Correctness with Reference | 结合参考答案判断代码正确性。 |
| `11-plan_adherence.py` | Plan Adherence | 判断 Agent 执行过程是否遵循既定计划。 |
| `12-openevals_evaluators.py` | 综合评估 | 使用多个 OpenEvals evaluator 对投顾助手做综合评估。 |
| `1-hybrid_wealth_advisor_langgraph_langsmith.py` | LangGraph Agent | 混合型财富管理投顾 Agent 示例，支持 LangSmith 追踪。 |
| `2-langsmith_testing_evaluation.py` | LangSmith 评估 | 创建测试数据集、运行批量测试并生成评估结果。 |

## 评估模式概览

### 1. Reference-based Evaluation

基于参考答案的评估。

典型输入：

```text
inputs + outputs + reference_outputs
```

适合：

- 正确性评估
- 代码正确性评估
- 有明确标准答案的问题

示例文件：

- `1-correctness.py`
- `10-code_correctness_with_reference.py`

### 2. Rubric-based LLM-as-Judge

不一定需要标准答案，而是让裁判模型根据评分标准判断输出质量。

典型输入：

```text
inputs + outputs
```

适合：

- 简洁性
- 相关性
- 安全性
- 是否有害
- 是否幻觉

示例文件：

- `2-conciseness.py`
- `3-answer_relevance.py`
- `7-toxicity.py`
- `8-hallucination.py`

### 3. RAG Evaluation

面向 RAG 检索增强生成的评估。

典型输入：

```text
inputs + context + outputs
```

或：

```text
context + outputs
```

适合：

- 检索内容是否相关
- 回答是否有上下文依据
- 回答是否对用户问题有帮助

示例文件：

- `4-rag_helpfulness.py`
- `5-rag_groundedness.py`
- `6-rag_retrieval_relevance.py`

### 4. Agent Evaluation

面向 Agent 工作流、工具调用或执行轨迹的评估。

适合：

- 判断处理模式是否正确
- 判断是否遵循计划
- 判断输出是否满足业务目标
- 结合 LangSmith trace 做实验分析

示例文件：

- `1-hybrid_wealth_advisor_langgraph_langsmith.py`
- `2-langsmith_testing_evaluation.py`
- `11-plan_adherence.py`
- `12-openevals_evaluators.py`

## 环境要求

- Python 3.10+
- DashScope API Key
- 可选：LangSmith API Key

## 安装依赖

建议在虚拟环境中安装依赖。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置环境变量

### DashScope

所有使用通义千问模型的脚本都需要配置 `DASHSCOPE_API_KEY`。

Windows PowerShell：

```powershell
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
```

macOS / Linux：

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### LangSmith 可选配置

如果要运行 LangSmith 数据集、实验追踪和评估示例，需要配置：

Windows PowerShell：

```powershell
$env:LANGSMITH_API_KEY="your-langsmith-api-key"
$env:LANGCHAIN_TRACING_V2="true"
$env:LANGCHAIN_PROJECT="openevals-wealth-advisor"
```

macOS / Linux：

```bash
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="openevals-wealth-advisor"
```

## 运行示例

运行单个 OpenEvals 示例：

```bash
python 1-correctness.py
python 2-conciseness.py
python 5-rag_groundedness.py
```

运行综合 evaluator 示例：

```bash
python 12-openevals_evaluators.py
```

运行 LangGraph 投顾 Agent：

```bash
python 1-hybrid_wealth_advisor_langgraph_langsmith.py
```

运行 LangSmith 测试与评估：

```bash
python 2-langsmith_testing_evaluation.py
```

## 常见评估输入字段

| 字段 | 含义 |
| --- | --- |
| `inputs` | 用户输入、问题或任务描述。 |
| `outputs` | 待评估的模型回答或代码输出。 |
| `reference_outputs` | 标准答案或参考输出，用于 correctness 类评估。 |
| `context` | RAG 检索得到的上下文资料。 |
| `trajectory` | Agent 执行轨迹、步骤或计划执行记录。 |

## OpenEvals 与 LangSmith 的关系

- `OpenEvals` 提供 evaluator，也就是具体的评估函数。
- `LangSmith` 提供数据集、实验运行、trace 记录和结果对比。
- 两者可以组合使用：用 LangSmith 管理实验，用 OpenEvals 判断输出质量。

简单理解：

```text
OpenEvals = 怎么评
LangSmith = 怎么批量跑、记录、对比
```

## 注意事项

- 示例中的裁判模型使用 `qwen-turbo`，结果会受模型能力、prompt 和网络状态影响。
- LLM-as-Judge 不是绝对客观评分，建议结合人工抽检和多维度 evaluator。
- RAG groundedness 只判断回答是否被给定上下文支撑，不负责判断外部世界事实是否真实。
- 请不要把真实 API Key、客户隐私数据或生产业务数据提交到 GitHub。
- 财富管理投顾示例仅用于技术演示，不构成投资建议。

