# Hybrid Wealth Advisor Qwen Agent with Langfuse

一个基于 `qwen-agent` 和阿里云 DashScope 通义千问模型构建的混合型财富管理投顾 AI 助手示例。项目包含普通版 Agent 和集成 Langfuse 可观测性的 Agent，用于演示智能投顾场景中的工具调用、客户画像注入、WebUI 交互以及 LLM 调用追踪。

## 项目亮点

- **混合智能体架构**：结合反应式响应与深思熟虑式分析，适配行情查询、资产配置、退休规划、教育金规划等不同复杂度的问题。
- **客户画像驱动**：内置两类示例客户画像，系统提示词会根据客户风险偏好、投资目标和资产配置动态生成。
- **工具调用示例**：通过 `qwen-agent` 工具机制注册上证指数查询工具，演示 Agent 如何调用外部工具获取结构化信息。
- **WebUI 交互体验**：使用 `qwen_agent.gui.WebUI` 快速启动浏览器聊天界面。
- **Langfuse 可观测性**：增强版脚本支持记录查询、响应、客户 metadata 和调用耗时，便于调试、评估和监控 LLM 应用。

## 目录结构

```text
.
├── 1-hybrid_wealth_advisor_qwen_agent.py
├── 1-hybrid_wealth_advisor_qwen_agent_langfuse.py
├── requirements.txt
└── readme.MD
```

| 文件 | 说明 |
| --- | --- |
| `1-hybrid_wealth_advisor_qwen_agent.py` | 基础版财富管理投顾 Agent，支持 Web GUI 和终端 TUI。 |
| `1-hybrid_wealth_advisor_qwen_agent_langfuse.py` | 集成 Langfuse 追踪的增强版 Agent，推荐用于调试和评估。 |
| `requirements.txt` | Python 运行依赖。 |

## 运行环境

- Python 3.10+
- DashScope API Key
- 可选：Langfuse Cloud 或自建 Langfuse 实例

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd test-evaluation-langfuse
```

如果该目录是某个大仓库中的子目录，请进入本目录后继续执行后续命令。

### 2. 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

至少需要配置 DashScope API Key：

Windows PowerShell：

```powershell
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
```

macOS / Linux：

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

如需启用 Langfuse 追踪，请继续配置：

Windows PowerShell：

```powershell
$env:LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
$env:LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
$env:LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

macOS / Linux：

```bash
export LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
export LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

> 如果使用自建 Langfuse，请将 `LANGFUSE_BASE_URL` 替换为你的服务地址。

## 使用方式

### 运行基础版 Agent

```bash
python 1-hybrid_wealth_advisor_qwen_agent.py
```

启动后可以选择：

1. Web 图形界面模式
2. 终端交互模式

### 运行 Langfuse 增强版 Agent

```bash
python 1-hybrid_wealth_advisor_qwen_agent_langfuse.py
```

增强版默认启动 Web 图形界面。如果已配置 Langfuse 密钥，脚本会自动记录完整查询链路和关键 metadata；如果未配置，仍可作为普通 Agent 运行。

## 示例问题

你可以在 WebUI 或终端中尝试以下问题：

- 今天上证指数的表现如何？
- 我的投资组合中科技股占比是多少？
- 根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？
- 考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。
- 我想为子女准备教育金，请帮我设计一个 10 年期的投资计划。

## 客户画像示例

项目内置两个模拟客户：

| 客户 | 风险偏好 | 投资期限 | 典型目标 |
| --- | --- | --- | --- |
| `customer1` | 平衡型 | 中期 | 退休规划、子女教育金 |
| `customer2` | 进取型 | 长期 | 财富增长、资产配置多元化 |

Agent 会根据客户画像动态调整回答角度，例如风险提示、资产配置建议和投资期限假设。

## Langfuse 追踪内容

增强版脚本会在可用时记录：

- 用户查询内容
- Agent 最终回答
- 客户 ID
- 风险偏好
- 投资期限
- 资产规模
- 查询长度
- 响应长度
- 调用耗时
- 是否发生错误

这适合用于：

- 调试 Prompt 和工具调用
- 对比不同客户画像下的响应质量
- 观察 LLM 应用延迟与异常
- 为后续评估集建设提供 trace 数据

## 注意事项

- 示例中的上证指数查询工具返回的是模拟数据，不代表真实行情。
- 本项目仅用于技术演示，不构成任何投资建议。
- 请勿将真实 API Key、客户隐私信息或账户数据提交到 GitHub。
- 如果 WebUI 启动失败，请先检查网络连接、DashScope API Key 和依赖安装情况。
