# LangChain 基础使用示例

基于 LangChain 生态的基础用法实践，使用通义千问（Qwen）大模型，展示各种 Chain 和 Agent 模式。

## 示例说明

| 文件 | 主题 | 内容 |
|---|---|---|
| `1-LLMChain.py` | Prompt + LLM 链 | 使用 `PromptTemplate` + `Tongyi` 模型，通过 `|` 管道符组合成链，根据产品名称生成公司名 |
| `2-LLMChain.py` | Agent + 外部工具 | 使用 `ChatTongyi` + `create_agent` + SerpAPI 搜索工具 + 自定义计算器，完成多步推理任务 |
| `3-LLMChain.py` | Agent + 自定义工具 | 基于预定义产品知识库的 Agent，演示自定义 `@tool` 的定义与调用 |
| `4-ConversationChain.py` | 带记忆的对话链 | 使用 `RunnableWithMessageHistory` + `MessagesPlaceholder` + `InMemoryChatMessageHistory` 实现多轮对话记忆 |
| `5-product_llm.py` | 交互式产品 Agent | 循环交互的产品问答 Agent，支持产品信息查询和公司介绍 |

## 安装依赖

```bash
pip install -r requirements.txt
```

需要设置环境变量 `DASHSCOPE_API_KEY`（通义千问 API Key）。

## 运行

```bash
python 1-LLMChain.py         # Prompt + LLM 链
python 2-LLMChain.py         # Agent + SerpAPI（需 SERPAPI_API_KEY）
python 3-LLMChain.py         # Agent + 自定义工具
python 4-ConversationChain.py # 带记忆的对话链
python 5-product_llm.py      # 交互式产品 Agent
```
