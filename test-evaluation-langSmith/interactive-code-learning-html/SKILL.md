---
name: interactive-code-learning-html
description: >-
  生成交互式代码学习 HTML 页面。适用于用户希望用高层视角学习代码，
  同时通过点击流程图节点查看真实代码、实现范式、输入输出和代码解读的场景。
  默认读取目标代码文件，按 code-learning-overview 的方法分析结构与关键节点，
  并在源文件同目录生成同名 .html 文件。
---

# 交互式代码学习 HTML

当用户想把某个代码文件做成“可点击、可导航、可学习”的 HTML 页面时，使用这个 skill。

本 skill 复用 `code-learning-overview` 的分析思想，但输出形式不是普通文本，而是一个交互式单文件 HTML。

---

## 核心目标

生成一个面向代码学习的交互式页面，把“高层地图”和“关键实现代码”连接起来。

页面应帮助用户：

- 先理解文件职责、主流程和思维模型
- 再通过流程图定位关键函数 / 节点
- 点击节点查看真实代码、实现方式和解读
- 按推荐阅读路径高效学习代码
- 避免陷入逐行阅读，也避免停留在过度抽象的总结

---

## 适用场景

当用户表达或暗示以下诉求时，使用本 skill：

- “帮我给这个文件生成一个代码学习 HTML。”
- “按那个交互式页面生成。”
- “生成可点击流程图的代码学习页面。”
- “把这个文件做成 onboarding html。”
- “用 skill 分析代码并生成 HTML。”
- “生成同目录的代码学习 dashboard。”
- “我想点击流程图节点，右侧看到具体代码和解释。”

---

## 输出文件规则

默认输出文件必须满足：

- 与源代码文件在同一目录
- 使用同样的文件名主体
- 扩展名改为 `.html`

例如：

```text
输入: test-evaluation-langSmith/1-hybrid_wealth_advisor_langgraph_langsmith.py
输出: test-evaluation-langSmith/1-hybrid_wealth_advisor_langgraph_langsmith.html
```

除非用户明确指定其他输出路径，否则不要改变这个规则。

---

## 工作流程

执行本 skill 时，按以下步骤操作：

1. 读取目标代码文件。
2. 读取本 skill 目录下的 HTML 模板文件：`template.html`。
3. 判断文件类型和技术栈。
4. 按 code-learning-overview 的方法建立学习地图。
5. 抽取关键函数、类、状态、prompt、框架调用和流程关系。
6. 识别每个关键节点的实现范式。
7. 为每个核心节点准备真实代码片段和解释。
8. 组装 codeData JSON 数据。
9. 基于 `template.html` 替换占位符生成最终 HTML。
10. 写入源文件同目录下的同名 .html 文件。
11. 告诉用户输出路径和页面包含的主要内容。

---

## 固定 HTML 模板

本 skill 配套一个固定模板文件：

```text
C:\Users\Administrator\.cursor\skills\interactive-code-learning-html\template.html
```

生成 HTML 时必须优先使用这个模板，不要每次从零重写页面结构。

模板使用以下占位符：

- `__PAGE_TITLE__`：HTML 页面标题
- `__CODE_DATA_JSON__`：页面数据对象，必须是可直接嵌入 JavaScript 的 JSON

生成流程：

1. 读取 `template.html`。
2. 根据目标代码文件生成 `codeData` JSON。
3. 将 `__PAGE_TITLE__` 替换成页面标题。
4. 将 `__CODE_DATA_JSON__` 替换成完整 JSON。
5. 输出到源代码文件同目录同名 `.html`。

除非用户明确要求重新设计 UI，否则不要改模板结构，只填充数据。

---

## 必须识别的实现范式

每个核心节点都必须标注“实现范式”。

常见分类：

### 原生 Python

普通函数、字典处理、条件判断、循环、异常处理等。

### LangChain

出现以下模式时标记为 LangChain：

```python
ChatPromptTemplate.from_template(...)
prompt | llm | JsonOutputParser()
prompt | llm | StrOutputParser()
chain.invoke(...)
```

### LangGraph

出现以下模式时标记为 LangGraph：

```python
StateGraph(...)
workflow.add_node(...)
workflow.add_edge(...)
workflow.add_conditional_edges(...)
workflow.set_entry_point(...)
workflow.compile()
agent.invoke(...)
```

### LangSmith

出现以下模式时标记为 LangSmith：

```python
RunnableConfig(...)
tags=[...]
metadata={...}
run_name=...
LANGCHAIN_TRACING_V2
LANGCHAIN_PROJECT
evaluate(...)
RunEvaluator
Client(...)
```

### Prompt 模板

大写字符串常量，尤其是用于 LLM 的模板，例如：

```python
ASSESSMENT_PROMPT = """..."""
```

### 工程胶水 / 环境配置

例如：

```python
load_dotenv(...)
importlib.util.spec_from_file_location(...)
warnings.filterwarnings(...)
os.getenv(...)
```

---

## HTML 页面结构

生成的 HTML 必须是单文件页面，优先使用：

- 内联 CSS
- 内联 JavaScript
- 不依赖 npm
- 不依赖构建工具
- 能直接双击打开

页面结构至少包含以下区域。

### 1. 顶部标题区

展示：

- 页面标题
- 文件名
- 文件路径
- 一句话定位
- 技术栈标签

### 2. 左侧学习导航区

包含 tab：

1. 概述
2. 流程图
3. 思维模型
4. 阅读路径
5. 常用任务
6. 提示词
7. 状态 / 数据

### 3. 中间流程图区

展示可点击节点图。

要求：

- 节点可点击
- 节点颜色区分类型
- 点击节点后更新右侧详情
- 对 LangGraph 文件，应尽量根据 `add_node`、`add_edge`、`add_conditional_edges` 识别真实控制流
- 不要简单按函数定义顺序伪造流程

### 4. 右侧节点详情区

点击节点后，展示：

- 节点名 / 函数名
- 分类
- 实现范式
- 职责
- 输入
- 输出
- 在整体流程中的位置
- 关键代码片段
- 代码逻辑解读
- 学习重点
- 可暂时忽略的细节
- 相关 prompt，如果存在

---

## 节点分类

节点可使用以下类型：

- `Entry`：入口点
- `Core`：核心业务 / 核心流程节点
- `Utility`：辅助函数 / 工具函数
- `External`：外部框架或平台能力
- `Prompt`：提示词模板
- `State`：状态 / 数据模型
- `Glue`：环境配置或工程胶水

颜色建议：

- Entry：青绿色
- Core：红色 / 粉色
- Utility：黄色
- External：蓝色
- Prompt：紫色
- State：绿色
- Glue：灰色

---

## 节点详情数据结构建议

HTML 内部可以使用类似结构：

```javascript
const codeData = {
  file: {
    name: "xxx.py",
    path: "...",
    role: "...",
    frameworks: ["Python", "LangGraph", "LangChain", "LangSmith"]
  },
  nodes: {
    run_wealth_advisor: {
      name: "run_wealth_advisor",
      type: "Entry",
      paradigm: "原生 Python + LangGraph invoke",
      role: "程序入口，构造初始 state 并启动图执行",
      input: "user_query, customer_id",
      output: "最终 state / final_response",
      code: "...真实代码片段...",
      explanation: "...代码逻辑解读...",
      learning: "...学习重点...",
      prompt: null
    }
  },
  edges: [
    ["run_wealth_advisor", "create_wealth_advisor_workflow"],
    ["assess_query", "reactive_processing"],
    ["assess_query", "collect_data"]
  ]
}
```

---

## 各 Tab 内容要求

### 概述

必须回答：

- 这个文件是做什么的？
- 属于什么类型：业务逻辑、流程编排、评估、追踪、工具还是胶水？
- 用了哪些主要框架？
- 入口函数是什么？
- 最值得学习的点是什么？

### 流程图

必须展示主流程。

对于 LangGraph 文件，优先从真实图结构提取：

- `set_entry_point`
- `add_node`
- `add_edge`
- `add_conditional_edges`
- `compile`

### 思维模型

用一个比喻帮助用户理解系统。

例如：

```text
这个文件像一个状态驱动的工作流机器。
用户输入被放进 state，state 在多个节点之间流动，
每个节点读取一部分 state，再写入新的字段，
最后由图结构决定下一步去哪里。
```

### 阅读路径

给出具体阅读顺序。

要求：

- 不要让用户从第一行开始读
- 每一步必须说明为什么先看这里
- 每一步对应具体函数 / 类 / 节点

### 常用任务

按学习目标组织快捷入口，例如：

- 我想看入口在哪里
- 我想看工作流怎么装配
- 我想看路由怎么决定
- 我想看 LLM 调用链怎么写
- 我想看 LangSmith 怎么接入
- 我想看最终回答在哪里生成

点击任务卡片后应跳转或展示对应节点。

### 提示词

如果代码包含 prompt，必须列出：

- prompt 名称
- 使用位置
- 输入变量
- 输出目标
- 作用
- prompt 片段或完整文本

不要默认把所有 prompt 展开到主流程里；prompt 应该作为单独学习视角。

### 状态 / 数据

如果代码包含状态类、TypedDict、BaseModel 或关键数据结构，必须说明：

- 字段名
- 存什么
- 为什么重要
- 哪些节点依赖它

---

## 代码展示与解读要求

不能只生成地图层。

每个核心节点必须有：

1. 真实代码片段
2. 实现范式说明
3. 输入输出说明
4. 代码逻辑解读
5. 学习重点

代码片段可以是完整函数，也可以是函数中最关键的部分。

不要把整个文件无差别塞进 HTML；只收录学习价值最高的关键代码。

---

## 压缩与取舍规则

默认收录：

- 入口函数
- workflow / graph 装配函数
- 路由函数
- 核心处理函数
- 关键状态类 / 数据模型
- 关键 prompt
- LangSmith / 评估 / tracing 接入点

默认不重点收录：

- 普通 import
- 重复性异常包装
- CLI 输入输出
- debug print
- 兼容性补丁
- 临时测试代码

这些可以放在“可暂时忽略”或 Glue 分类里。

---

## 生成后回复用户

完成后，简洁告诉用户：

- 已生成 HTML 文件
- 输出路径
- 页面包含哪些主要视图
- 建议从哪个 tab 开始看

示例：

```text
已生成交互式代码学习页面：
`path/to/file.html`

页面包含：概述、流程图、思维模型、阅读路径、常用任务、提示词、状态/数据和节点代码详情。
建议先看“概述”和“思维模型”，再按“阅读路径”点击节点学习。
```

---

## 成功标准

本 skill 成功时，用户应该能够：

- 打开 HTML 后快速知道文件职责
- 看懂主流程和关键分支
- 点击节点看到真实代码和解释
- 知道每段核心代码属于原生 Python / LangChain / LangGraph / LangSmith 的哪一类实现
- 按推荐路径学习，而不是从第一行读到最后一行
- 同时避免“过度抽象”和“陷入细节”
