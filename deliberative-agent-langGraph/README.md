# case-deliberative-agent-langGraph

多阶段投研智能体示例目录，当前包含三类实现：

- 基于 `LangGraph` 的阶段化研究工作流
- 基于 `qwen-agent` 的工具调用式投研智能体
- 基于 `LangGraph` 的混合式财富顾问智能体

这个目录主要用于演示如何把“感知—建模—推理—决策—输出”这类复杂业务流程组织成可执行智能体系统，并对比不同 Agent 框架与工作流设计方式。

---

## 目录结构

```text
case-deliberative-agent-langGraph/
├─ deliberative_research_langgraph.py
├─ deliberative_research_qwen_agent.py
├─ hybrid_wealth_advisor_langgraph.py
├─ requirements.txt
├─ research_report_20260422_221344.txt
└─ README.md
```

---

## 这个目录包含什么

### 1. `deliberative_research_langgraph.py`

基于 `LangGraph` 构建的深思熟虑型投研工作流。

实现思路是将投研过程拆成多个阶段节点，并通过状态对象在各节点间传递中间结果。

当前核心阶段包括：

1. 感知（Perception）
2. 建模（Modeling）
3. 推理（Reasoning）
4. 决策（Decision）
5. 报告（Report）

该实现适合用于说明：

- 如何定义 `StateGraph`
- 如何在节点中调用 LLM
- 如何通过状态对象保存阶段产物
- 如何把复杂分析流程拆解为多阶段图结构
- 固定边与条件边的设计差异

### 2. `deliberative_research_qwen_agent.py`

基于 `qwen-agent` 实现的深思熟虑型投研智能体。

这一版本将投研能力封装为工具，再通过智能体调度这些工具完成多阶段分析。

该实现适合用于说明：

- 如何在 `qwen-agent` 中注册工具
- 如何组织系统提示词与工具描述
- 如何让 Agent 决定调用哪个分析工具
- 如何生成完整研究报告

### 3. `hybrid_wealth_advisor_langgraph.py`

基于 `LangGraph` 构建的混合式财富顾问智能体。

这一实现结合了两种处理模式：

- `reactive`：对需要快速响应的问题走工具调用链路
- `deliberative`：对需要深度分析的问题走多阶段分析链路

该实现适合用于说明：

- 如何在同一工作流中融合 reactive 与 deliberative 两类模式
- 如何用 `messages` 状态维护工具调用消息历史
- 如何通过 `ToolNode` 构建工具调用循环
- 如何按用户问题类型动态选择处理分支

---

## 核心业务思路

这个目录中的示例都围绕同一个主题：

- 先理解用户问题或研究主题
- 再组织信息、分析、推理和建议生成
- 最终输出一份结构化或可直接阅读的结论

其中不同脚本主要对比的是“如何组织智能体流程”：

- 是用状态图显式分阶段推进
- 还是用工具调用式 Agent 自主调度
- 还是将快速响应与深度分析结合成混合结构

---

## LangGraph 版说明

`deliberative_research_langgraph.py` 使用的核心能力包括：

- `StateGraph`
- `ChatPromptTemplate`
- `Tongyi`
- `JsonOutputParser`
- `PydanticOutputParser`
- `StrOutputParser`

其中每个节点会：

- 读取当前状态
- 组装提示词
- 调用 LLM
- 解析输出
- 返回更新后的状态

这类写法适合教学和流程分析，因为每一步的状态更新都比较清晰。

---

## qwen-agent 版说明

`deliberative_research_qwen_agent.py` 使用的核心能力包括：

- `Assistant`
- `WebUI`
- `BaseTool`
- `register_tool`
- `dashscope`

这一版本更偏向“工具化智能体”思路。

系统会：

- 根据用户研究需求选择工具
- 优先通过 `complete_analysis` 工具完成全流程分析
- 必要时再分步骤调用各阶段工具

该版本更适合展示“工具调度式 Agent”的实现方式。

---

## 混合式财富顾问版说明

`hybrid_wealth_advisor_langgraph.py` 使用的核心能力包括：

- `StateGraph`
- `ToolNode`
- `bind_tools(...)`
- `messages: Annotated[List[BaseMessage], add_messages]`
- 条件边 `add_conditional_edges(...)`

这一版本的特点是：

1. 先评估用户问题属于快速响应还是深度分析
2. 如果属于 reactive 场景，则进入工具调用循环
3. 如果属于 deliberative 场景，则进入多阶段分析链路
4. 最终统一输出回答或建议

这个案例适合用于说明：

- 混合型 Agent 架构
- 消息历史在工具调用中的作用
- LangGraph 中状态驱动的分支控制

---

## 依赖安装

建议在独立虚拟环境中安装依赖。

在仓库根目录执行：

```bash
pip install -r case-deliberative-agent-langGraph/requirements.txt
```

当前依赖：

- `dashscope==1.22.1`
- `langchain==1.0.3`
- `langchain-community==0.4.1`
- `langchain-core==1.0.3`
- `langgraph==1.0.2`
- `matplotlib==3.10.7`
- `pydantic==2.11.7`
- `qwen-agent==0.0.25`

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

### 运行 LangGraph 投研版

```bash
python case-deliberative-agent-langGraph/deliberative_research_langgraph.py
```

### 运行 qwen-agent 投研版

```bash
python case-deliberative-agent-langGraph/deliberative_research_qwen_agent.py
```

### 运行混合式财富顾问版

```bash
python case-deliberative-agent-langGraph/hybrid_wealth_advisor_langgraph.py
```

具体交互形式取决于脚本入口设计和本地环境配置。

---

## 适合用在什么场景

这个目录适合作为以下场景的参考：

- 学习 Deliberative Agent 的阶段化设计
- 对比 LangGraph 与 qwen-agent 两种实现方式
- 学习如何将投研流程拆解成多阶段智能体节点
- 学习结构化输出解析与报告生成
- 学习混合型 Agent 的分支路由与工具调用循环
- 作为金融问答、投研分析、财富顾问类 Agent 的原型起点

---

## 已知限制

当前示例更偏向教学和原型验证，存在以下简化点：

1. 研究数据主要依赖模型知识和提示词组织，未接入真实外部数据源
2. 状态流和工具流主要用于演示结构，不代表完整生产系统设计
3. 报告质量受模型能力、提示词和输入主题影响较大
4. 目录中可能出现本地生成的研究报告文件，这类产物通常不应纳入版本控制
5. 某些路由或状态字段可能同时承担“当前阶段记录”和“下一跳控制”两种职责，更适合用于教学分析而不是直接生产化复用

---

## 后续可优化方向

如果需要进一步沉淀为可复用案例，可以继续做这些改进：

1. 为 LangGraph 版补充真正生效的条件路由
2. 将阶段结果落盘为结构化 JSON，便于复盘和评测
3. 接入外部检索、财经数据接口或知识库
4. 给 qwen-agent 版补充更清晰的工具调用日志
5. 给混合式财富顾问版补充 Prompt 调试与中间状态可视化
6. 引入评估机制，对不同方案输出进行自动比较
7. 补充示例输入与示例报告输出，方便快速验证

---

## 版本控制建议

建议保留：

- `deliberative_research_langgraph.py`
- `deliberative_research_qwen_agent.py`
- `hybrid_wealth_advisor_langgraph.py`
- `requirements.txt`
- 本 README

建议排除：

- 本地生成的研究报告文件
- 本地调试时产生的临时文件
- 与个人环境或私有密钥相关的本地配置文件

例如：

- `research_report_*.txt`

这个目录适合作为以下能力的参考案例：

- `build-deliberative-agent`
- `langgraph-research-workflow`
- `qwen-agent-tool-orchestration`
- `hybrid-wealth-advisor-agent`
- `multi-stage-investment-research-agent`
