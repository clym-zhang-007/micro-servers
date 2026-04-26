# case-react-agent-langchain

使用 `LangChain Classic ReAct Agent + ChatTongyi` 构建的最小私募基金问答示例。

这个目录主要用于演示：

- 如何为一个问答场景定义本地工具
- 如何使用 ReAct 模式让 Agent 自主决定调用哪个工具
- 如何观察 Agent 的 `Thought / Action / Observation / Final Answer` 链路
- 如何打印每一轮实际发送给 LLM 的 Prompt，辅助理解 prompt 拼装方式

当前实现刻意采用 `langchain_classic.agents`，以避开当前 `langchain.agents.create_agent` 在部分版本组合下的兼容问题。

---

## 目录结构

```text
case-react-agent-langchain/
├─ fund_qa_langchain.py
├─ requirements.txt
└─ README.md
```

---

## 示例实现说明

`fund_qa_langchain.py` 提供了一个面向私募基金规则问答的最小 Agent 示例。

内置了一份简化规则库 `FUND_RULES_DB`，并定义了 3 个工具：

- `search_rules_by_keywords`
  - 根据关键词搜索规则
- `search_rules_by_category`
  - 按类别查询规则
- `answer_question`
  - 直接对用户问题做匹配并返回最相关规则

在 Agent 层，脚本使用：

- `ChatTongyi`
- `create_react_agent`
- `AgentExecutor`

来构造一个具备 ReAct 循环的问答助手。

---

## ReAct 链路说明

运行时，Agent 会按照如下模式工作：

1. 接收用户问题
2. 根据 Prompt 先生成 `Thought`
3. 决定要调用的工具 `Action`
4. 传入工具参数 `Action Input`
5. 读取工具返回结果 `Observation`
6. 将已有轨迹继续拼接回 Prompt
7. 再次调用 LLM，直到生成 `Final Answer`

这类流程通常被称为：

- 思考（Thought）
- 执行（Action）
- 观察（Observation）
- 再思考（Thought）

也就是常见的 ReAct 智能体模式。

---

## Prompt 调试能力

当前脚本包含调试输出，便于观察 Agent 是如何工作的。

主要包括两部分：

### 1. 首轮 Prompt 打印

在每次用户输入后，会先打印起始 Prompt，帮助理解：

- `{tools}` 是如何展开的
- `{tool_names}` 最终长什么样
- `{input}` 是如何注入的
- `{agent_scratchpad}` 初始时为什么为空

### 2. 每一轮 LLM 调用打印

通过 `PromptDebugHandler`，脚本会打印：

- 每一轮真正发送给 LLM 的 Prompt
- 每一轮 LLM 的文本输出

因此这个示例不仅能跑通 Agent，还适合用来学习：

- Agent Prompt 是如何拼装的
- ReAct 轨迹是如何逐轮累积的
- 为什么多轮工具调用会让 Prompt 越来越长

---

## 中文分词说明

在 `answer_question` 工具中，脚本使用 `jieba` 对中文问题和规则文本做分词。

相比直接使用 `split()`：

- 更适合中文场景
- 更容易提取有效词项
- 更适合做简单的规则匹配和交集计算

当前分词逻辑只是一个轻量示例，适合教学和原型验证。

---

## 依赖安装

建议在独立虚拟环境中安装依赖。

在仓库根目录执行：

```bash
pip install -r case-react-agent-langchain/requirements.txt
```

当前依赖：

- `langchain==1.0.3`
- `langchain_community==0.4.1`
- `jieba==0.42.1`

如果使用 `ChatTongyi`，还需要确保本机环境已具备对应运行条件，并可正常访问 DashScope 服务。

---

## 环境变量

运行前需要配置：

```bash
DASHSCOPE_API_KEY
```

例如在 PowerShell 中：

```powershell
$env:DASHSCOPE_API_KEY="你的_api_key"
```

如果没有配置该变量，模型调用将无法正常完成。

---

## 运行方式

在仓库根目录执行：

```bash
python case-react-agent-langchain/fund_qa_langchain.py
```

启动后可以直接输入问题，例如：

- `私募基金的合格投资者标准是什么？`
- `监管规定有哪些？`
- `风险准备金要求是什么？`

输入以下内容可退出：

- `退出`
- `exit`
- `quit`

---

## 适合用在什么场景

这个目录适合作为以下场景的参考：

- 学习 LangChain 中 ReAct Agent 的最小实现
- 观察工具调用型 Agent 的中间轨迹
- 学习 Prompt 模板变量是如何被自动注入的
- 理解 `agent_scratchpad` 在多轮推理中的作用
- 作为中文规则库问答原型的起点

---

## 已知限制

当前示例是教学型最小实现，存在一些有意保留的简化点：

1. 规则库写死在 Python 文件中，未接入外部知识库
2. 检索逻辑较轻量，未使用向量检索或 BM25
3. 工具数量较少，适合演示但不适合复杂生产场景
4. 调试输出较多，更适合学习和排查，不适合直接作为生产终端体验
5. 当前采用 `langchain_classic.agents`，主要是为兼容现有版本链路

---

## 后续可优化方向

如果需要把这个目录继续沉淀为可复用示例，可以继续做这些改进：

1. 将规则库抽离到独立文件或数据库
2. 增加更多类别与问题样本
3. 引入向量检索或 BM25 提升召回效果
4. 将 Prompt 调试输出做成可开关模式
5. 增加更清晰的日志结构，区分 Prompt、工具调用和最终回答
6. 为不同 LLM 后端适配统一接口

---

## 版本控制建议

建议保留：

- `fund_qa_langchain.py`
- `requirements.txt`
- 本 README

建议排除：

- 本地环境相关缓存
- 个人机器上的临时调试文件
- 与密钥、私有配置相关的本地文件

这个目录适合作为以下能力的参考案例：

- `build-react-agent-demo`
- `observe-react-agent-prompting`
- `tool-calling-agent-with-debug-print`
- `langchain-classic-react-agent`
