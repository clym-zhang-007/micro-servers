# -*- coding: utf-8 -*-
"""多文件 RAG 问答示例 - 基于 Qwen Agent 框架

功能特性:
    - 自定义 AI 绘画工具 (my_image_gen)，根据文本描述生成图像
    - 代码执行能力 (code_interpreter)，支持图像处理与展示
    - 多文件检索 (RAG)：自动加载 docs/ 目录下所有文件作为知识库
    - 终端流式输出，支持实时查看模型回复

典型使用场景:
    - 基于本地文档的智能问答（如保险条款查询）
    - 图像生成与处理
    - 多工具协同的复杂任务

依赖:
    - qwen_agent
    - json5
    - urllib

运行方式:
    - 直接运行: python qwen-agent-multi-files.py
    - 环境变量: 需设置 DASHSCOPE_API_KEY
"""
import pprint
import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import os

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
        prompt = urllib.parse.quote(prompt)
        return json5.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False)


# 步骤 2：配置您所使用的 LLM。
llm_cfg = {
    # 使用 DashScope 提供的模型服务：
    'model': 'deepseek-v3',
    'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
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
import os
# 获取文件夹下所有文件
file_dir = os.path.join('./', 'docs')
files = []
if os.path.exists(file_dir):
    # 遍历目录下的所有文件
    for file in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file)
        if os.path.isfile(file_path):  # 确保是文件而不是目录
            files.append(file_path)
print('files=', files)

bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                files=files)

# 步骤 4：作为聊天机器人运行智能体。
messages = []  # 这里储存聊天历史。
query = "介绍下雇主责任险"
# 将用户请求添加到聊天历史。
messages.append({'role': 'user', 'content': query})
response = []
current_index = 0
for response in bot.run(messages=messages):
    if current_index == 0:
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
        #break

    current_response = response[0]['content'][current_index:]
    current_index = len(response[0]['content'])
    print(current_response, end='')
# 将机器人的回应添加到聊天历史。
#messages.extend(response)
