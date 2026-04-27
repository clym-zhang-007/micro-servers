import os
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent
import dashscope

# 从环境变量获取 dashscope 的 API Key
api_key = os.getenv('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

#   加载模型 (使用 ChatModel 以支持 tool calling)
#   一定要用 ChatTongyi。原因：

#   ChatTongyi 和 Tongyi 是两种不同的接口：

#   ┌────────────┬────────────────────────┬───────────────────┬─────────────────┐
#   │     类     │        接口类型        │ 支持 Tool Calling │    适用场景     │
#   ├────────────┼────────────────────────┼───────────────────┼─────────────────┤
#   │ ChatTongyi │ Chat 模型（消息对话）  │ ✅ 支持           │ Agent、多轮对话 │
#   ├────────────┼────────────────────────┼───────────────────┼─────────────────┤
#   │ Tongyi     │ 补全模型（completion） │ ❌ 不支持         │ 简单的文本生成  │
#   └────────────┴────────────────────────┴───────────────────┴─────────────────┘

#   create_agent 内部需要模型支持 tool calling 能力（让模型知道什么时候该调用工具、怎么调用），只有 ChatTongyi
#   提供了这个能力。如果用 Tongyi，会直接报错。

llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=api_key)

# 加载 serpapi 工具 -- 返回的是 Google 搜索结果，数据更丰富、更准确
tools = load_tools(["serpapi"])

# LangChain 1.x 新写法
agent = create_agent(llm, tools)

# 运行 agent
result = agent.invoke({"messages": [("user", "今天是几月几号?历史上的今天有哪些名人出生")]})
print(result["messages"][-1].content)
