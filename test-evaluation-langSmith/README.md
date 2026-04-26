# LangSmith / DeepEval 财富管理投顾助手评估示例

本目录演示如何使用 **LangGraph + LangSmith + DeepEval** 构建、追踪和评估一个混合型财富管理投顾 AI 助手。

项目包含一个基于 LangGraph 的 Hybrid Agent，并提供两套评估方式：

- **LangSmith 评估**：适合做链路追踪、数据集管理、实验记录和结果可视化。
- **DeepEval 评估**：适合做回答质量自动化测试，例如相关性、幻觉检测和自定义业务标准评估。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `1-hybrid_wealth_advisor_langgraph_langsmith.py` | 财富管理投顾 Hybrid Agent 主程序，集成 LangGraph 和 LangSmith tracing。 |
| `2-langsmith_testing_evaluation.py` | LangSmith 数据集创建与批量评估脚本。 |
| `3-run_langsmith_evaluation_example.py` | LangSmith 快速开始示例脚本。 |
| `deepeval_wealth_advisor.py` | DeepEval 自动化质量评估脚本。 |
| `requirements.txt` | 当前示例所需 Python 依赖。 |
| `.env` | 本地环境变量文件，建议只保留示例配置，不要提交真实 API Key。 |

## 环境要求

建议使用 Python 3.10+，推荐 Python 3.11。

```bash
python --version
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 conda：

```bash
conda create -n wealth-eval python=3.11 -y
conda activate wealth-eval
pip install -r requirements.txt
```

## 环境变量配置

本项目需要模型服务和评估平台相关 API Key。

### 1. DashScope / 通义千问

主智能体使用通义千问模型，需要配置：

```bash
DASHSCOPE_API_KEY=your-dashscope-api-key
```

Windows PowerShell 示例：

```powershell
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 2. LangSmith

如果需要启用 LangSmith tracing 和评估，需要配置：

```bash
LANGSMITH_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=wealth-advisor-hybrid-agent
```

Windows PowerShell 示例：

```powershell
$env:LANGSMITH_API_KEY="your-langsmith-api-key"
$env:LANGCHAIN_TRACING_V2="true"
$env:LANGCHAIN_PROJECT="wealth-advisor-hybrid-agent"
```

### 3. DeepEval / OpenAI

`deepeval_wealth_advisor.py` 中的 DeepEval 指标默认使用 OpenAI 模型作为评估器，需要配置：

```bash
OPENAI_API_KEY=your-openai-api-key
```

Windows PowerShell 示例：

```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
```

## 本地 `.env` 示例

可以在当前目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your-dashscope-api-key
LANGSMITH_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=wealth-advisor-hybrid-agent
OPENAI_API_KEY=your-openai-api-key
```

注意：不要把真实 API Key 提交到 Git。建议在 `.gitignore` 中忽略 `.env`。

## 运行主智能体

```bash
python 1-hybrid_wealth_advisor_langgraph_langsmith.py
```

启用 LangSmith 后，可以在 [LangSmith](https://smith.langchain.com) 中查看 Agent 执行 trace，包括 prompt、模型调用、中间状态、耗时等信息。

## 运行 LangSmith 评估

### 方式一：运行完整评估脚本

```bash
python 2-langsmith_testing_evaluation.py
```

该脚本会：

1. 检查 LangSmith 是否启用。
2. 创建或复用 LangSmith Dataset。
3. 写入测试用例。
4. 调用 `evaluate()` 执行批量评估。
5. 将实验结果上传到 LangSmith。

### 方式二：运行快速开始示例

```bash
python 3-run_langsmith_evaluation_example.py
```

该脚本会先运行单条测试，再询问是否执行完整评估。

## 运行 DeepEval 评估

```bash
python deepeval_wealth_advisor.py
```

也可以使用 DeepEval 的测试命令：

```bash
deepeval test run deepeval_wealth_advisor.py
```

DeepEval 脚本包含以下指标：

- `AnswerRelevancyMetric`：评估回答是否切题。
- `HallucinationMetric`：检测回答是否存在幻觉。
- `GEval`：自定义评估投资建议是否考虑客户风险偏好。

## LangSmith 与 DeepEval 的区别

| 工具 | 更适合做什么 |
| --- | --- |
| LangSmith | 链路追踪、数据集管理、实验对比、可视化调试。 |
| DeepEval | 回答质量自动化测试、CI 质量门禁、幻觉检测、自定义评分标准。 |

简单理解：

- **LangSmith 看过程**：帮助你分析 Agent 每一步为什么这样执行。
- **DeepEval 判质量**：帮助你判断最终回答是否达标。

## 测试用例说明

LangSmith 脚本中的测试用例主要分为三类：

- `REACTIVE_TEST_CASES`：简单查询、概念解释、快速响应类问题。
- `DELIBERATIVE_TEST_CASES`：投资规划、风险分析、长期建议类复杂问题。
- `EDGE_CASE_TEST_CASES`：空输入、超长输入等边界情况。

DeepEval 脚本中的测试用例主要关注：

- 回答是否与问题相关。
- 是否基于上下文回答。
- 是否考虑客户风险等级、投资期限和偏好。

## 常见问题

### 1. 提示 `DASHSCOPE_API_KEY` 未设置

请先配置通义千问 API Key：

```powershell
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 2. 提示 `LANGSMITH_API_KEY` 或 `LANGCHAIN_TRACING_V2` 未设置

如果要运行 LangSmith 评估，请配置：

```powershell
$env:LANGSMITH_API_KEY="your-langsmith-api-key"
$env:LANGCHAIN_TRACING_V2="true"
```

### 3. 提示 `OPENAI_API_KEY` 未设置

DeepEval 默认需要 OpenAI 作为评估模型，请配置：

```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
```

### 4. `deepeval` 找不到

安装依赖：

```bash
pip install -r requirements.txt
```

或单独安装：

```bash
pip install deepeval
```

## Git 提交注意事项

建议不要提交以下内容：

- 真实 `.env` 文件
- API Key
- 本地虚拟环境目录，例如 `.venv/`
- Python 缓存目录，例如 `__pycache__/`

如果需要提供环境变量模板，可以提交 `.env.example`，但不要包含真实密钥。

## 参考链接

- [LangSmith](https://smith.langchain.com)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://python.langchain.com)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [DashScope](https://help.aliyun.com/zh/dashscope/)
