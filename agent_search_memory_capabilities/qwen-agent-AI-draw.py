# -*- coding: utf-8 -*-
"""AI搜索问答助手 - 基于 Qwen Agent 框架构建的智能 Agent

功能特性:
    - 自定义 AI 绘画工具 (my_image_gen)，根据文本描述生成图像
    - 代码执行能力 (code_interpreter)，支持图像处理与展示
    - 本地文档检索，基于 RAG 的知识问答
    - 支持 WebUI 图形界面和终端 (TUI) 两种交互模式

典型使用场景:
    - 图像生成与处理
    - 基于本地文档的智能问答
    - 多工具协同的复杂任务

依赖:
    - qwen_agent
    - json5
    - urllib

运行方式:
    - 直接运行: python ai_bot-1.py (默认启动 WebUI)
    - 环境变量: 需设置 DASHSCOPE_API_KEY
"""
import pprint
import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import os
from qwen_agent.gui import WebUI


# 步骤 1（可选）：添加一个名为 `my_image_gen` 的自定义工具。
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    # `description` 用于告诉智能体该工具的功能。
    description = 'AI 绘画（图像生成）服务，输入文本描述，返回基于文本信息绘制的图像 URL。'
    # `parameters` 告诉智能体该工具有哪些输入参数。
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '期望的图像内容的详细描述',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        # `params` 是由 LLM 智能体生成的参数。
        prompt = json5.loads(params)['prompt']
        #    pollinations.ai 的图像生成 API 要求 prompt 作为 URL 路径的一部分，如果 prompt 中包含空格、中文、特殊字符等，直接拼接到
        #    URL 中会导致请求失败。urllib.parse.quote 会将这些字符转换为安全的 URL 编码格式：
        #   - 一只猫 → %E4%B8%80%E5%8F%AA%E7%8C%AB
        #   - a cat → a%20cat（空格转义）
        prompt = urllib.parse.quote(prompt)
        return json5.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False
        )


def init_agent_service():
    """初始化助手服务"""
    # 步骤 2：配置您所使用的 LLM。
    llm_cfg = {
        # 使用 DashScope 提供的模型服务：
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),  # 从环境变量获取API Key
        'generate_cfg': {
            'top_p': 0.8
        }
    }

    # 步骤 3：创建一个智能体。这里我们以 `Assistant` 智能体为例，它能够使用工具并读取文件。
    system_instruction = '''你是一个乐于助人的AI助手。
在收到用户的绘画请求后，你应该：
- 调用 my_image_gen 工具生成图像，得到图像 URL。
- 用 Markdown 图片格式展示图像：`![图像描述](图像URL)`。
- 如果需要对图像进行处理，使用 code_interpreter 下载并处理图像，用 `plt.show()` 展示。
你总是用中文回复用户。'''
    tools = ['my_image_gen', 'code_interpreter']  # `code_interpreter` 是框架自带的工具，用于执行代码。
    # 获取文件夹下所有文件
    file_dir = os.path.join(os.path.dirname(__file__), 'docs')
    files = []
    if os.path.exists(file_dir):
        # 遍历目录下的所有文件
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):  # 确保是文件而不是目录
                files.append(file_path)
    print('files=', files)

    #     当把文件列表传给 Assistant 后，框架会自动做两件事：
    #   - 建立索引：对文件内容进行分词/向量化，构建检索索引
    #   - 按需检索（RAG）：用户提问时，根据问题内容从文件中检索相关片段，注入到对话上下文中
    bot = Assistant(llm=llm_cfg,
                    system_message=system_instruction,
                    function_list=tools,
                    files=files)
    return bot


def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('user question: ')
                
                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                    
                # 构建消息
                messages.append({'role': 'user', 'content': query})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                current_index = 0
                first_chunk = True
                for response_chunk in bot.run(messages=messages):
                    if first_chunk:
                        # 尝试获取并打印召回的文档内容
                        if hasattr(bot, 'retriever') and bot.retriever:
                            print("\n===== 召回的文档内容 =====")
                            retrieved_docs = bot.retriever.retrieve(query)
                            if retrieved_docs:
                                for i, doc in enumerate(retrieved_docs):
                                    print(f"\n文档片段 {i+1}:")
                                    print(f"内容: {doc.page_content}")
                                    print(f"元数据: {doc.metadata}")
                            else:
                                print("没有召回任何文档内容")
                            print("===========================\n")
                        first_chunk = False

                    # The response is a list of messages. We are interested in the assistant's message.
                    if response_chunk and response_chunk[0]['role'] == 'assistant':
                        assistant_message = response_chunk[0]
                        new_content = assistant_message.get('content', '')
                        print(new_content[current_index:], end='', flush=True)
                        current_index = len(new_content)
                    
                    response = response_chunk
                
                print() # New line after streaming.

                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举3个典型门票查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '画一只在写代码的猫',
                '介绍下雇主责任险',
                '帮我画一个宇宙飞船，然后把它变成黑白的'
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）
    # app_tui() 