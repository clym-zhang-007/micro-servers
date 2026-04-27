import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import dashscope

# 从环境变量获取 dashscope 的 API Key
api_key = os.getenv('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# 加载模型
llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=api_key)

# 创建带历史记录的 prompt
# MessagesPlaceholder 的作用是在提示词模板中"挖一个坑"，留给外部代码在运行时动态填入一段消息列表。
# 实际调用时，你需要传入一个消息列表：
#   prompt.invoke({
#       "input": "今天天气如何？",
#       "history": [
#           HumanMessage("你好"),
#           AIMessage("你好！有什么我可以帮你的？"),
#       ]
#   })
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  # ← 这里挖了一个叫 history 的坑，像前端input的placeholder，占位需要被替换掉
    ("human", "{input}")
])

# 创建 chain
chain = prompt | llm

# 存储会话历史
# store 是一个简单的内存字典，用 session_id 作为 key 存储每个用户的对话历史
# 生产环境通常替换为数据库（如 Redis）以实现持久化
store = {}

def get_session_history(session_id: str):
    """
    获取或创建指定 session_id 的对话历史存储对象
    这个函数会被 RunnableWithMessageHistory 自动调用
    """
    if session_id not in store:
        # InMemoryChatMessageHistory 是 LangChain 提供的内存版历史记录器
        # 它实现了 BaseChatMessageHistory 接口，支持 add_messages() 和 get_messages()
        # 每个session_id 对应的历史记录都会被存在一个 InMemoryChatMessageHistory 实例中
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建带记忆的对话链
# RunnableWithMessageHistory 是一个"包装器"，它给普通的 chain 自动加上记忆能力：
# 所有的 add_messages 和 get_messages 都由 RunnableWithMessageHistory 内部实现，业务代码无需调用这两个函数去管理记忆
# 1. 执行前：自动从 get_session_history 取出历史消息，填入 prompt 中 MessagesPlaceholder("history") 的位置
# 2. 执行后：自动把本轮的 user 输入和 AI 回复追加到历史记录中，下次对话会包含这段记忆
conversation = RunnableWithMessageHistory(
    chain,                      # 要包装的原始 chain（prompt | llm）
    get_session_history,        # 获取历史记录器的回调函数
    input_messages_key="input",     # 告诉包装器：用户输入对应 prompt 中的哪个变量
    history_messages_key="history"  # 告诉包装器：历史消息对应 prompt 中的 MessagesPlaceholder 名称
)

config = {"configurable": {"session_id": "default"}}
# config 是 LangChain 的配置传递机制，"configurable" 下的参数会被透传给 get_session_history
# 这里 session_id="default" 表示所有对话共享同一个会话历史

# 第一轮对话
# 此时 store 为空，history 占位填入空列表，AI 只看到 system 提示词和用户输入
output = conversation.invoke({"input": "Hi!,我是 Clym"}, config=config)
print(output.content, "output1")

# 第二轮对话 (会记住上一轮)
# 此时 RunnableWithMessageHistory 自动从 store 中取出第一轮的对话历史，
# 拼到 prompt 的 history 位置，所以 AI 能"回忆"起第一轮用户说的名字
output = conversation.invoke({"input": "我叫什么？"}, config = config)
print(output.content, "output2")
