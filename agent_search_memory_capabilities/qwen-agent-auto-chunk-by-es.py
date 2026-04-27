# ai_bot-3.py
import os
from dotenv import load_dotenv
import asyncio
from typing import Optional
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import warnings
warnings.filterwarnings("ignore")

# 加载 .env 环境变量
load_dotenv()

def init_agent_service():
    """初始化具备 Elasticsearch RAG 能力的助手服务"""
    
    # 步骤 1: LLM 配置
    llm_cfg = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generate_cfg': {
            'top_p': 0.8
        }
    }

    # 步骤 2: RAG 配置 - 激活并配置 Elasticsearch 后端
    rag_cfg = {
        "rag_backend": "elasticsearch",  # 告诉 qwen-agent 使用 ES 作为检索后端
                                       # 不配则走框架内置的默认检索（内存/本地向量库）
        "es": {
            "host": "https://localhost",
            "port": 9200,
            "user": "elastic",
            "password": os.getenv('ES_PASSWORD', ''),  # 从环境变量读取
            "index_name": "my_insurance_docs_index"  # ES 中索引的名称
                                                    # Agent 会自动往这个索引写文档、查文档
        },
        "parser_page_size": 500 # 文档分块大小
    }

    # 步骤 3: 系统指令和工具
    system_instruction = '''你是一个基于本地知识库的AI助手。
请根据用户的问题，利用检索工具从知识库中查找最相关的信息，并结合这些信息给出专业、准确的回答。'''

    # 获取文件夹下所有文件
    file_dir = os.path.join(os.path.dirname(__file__), 'docs')
    files = []
    if os.path.exists(file_dir):
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):
                files.append(file_path)
    print('知识库文件列表:', files)

    # 步骤 4: 创建智能体实例
    # 通过 rag_cfg 参数传入我们的 ES 配置

    # 工作流程

    # 1. 初始化 Assistant 时，框架读取 files 参数里的文档
    # 2. 按 parser_page_size=500 把每个文档切成 500 字符的片段
    # 3. 把片段写入 ES 的 my_insurance_docs_index 索引
    # 4. 用户提问时，Agent 自动向该索引发检索请求，拿回相关片段后注入 LLM 上下文
    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        files=files,
        rag_cfg=rag_cfg
    )
    return bot

def main():
    """启动 Web 图形界面"""
    try:
        print("正在启动 AI 助手 Web 界面 (Elasticsearch 后端)...")
        bot = init_agent_service()
        chatbot_config = {
            'prompt.suggestions': [
                '介绍下雇主责任险',
                '雇主责任险和工伤保险有什么主要区别？',
                '介绍一下平安商业综合责任保险（亚马逊）的保障范围。',
                '施工保主要适用于哪些场景？',
            ]
        }
        WebUI(bot, chatbot_config=chatbot_config).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {e}")
        print("请检查网络连接、API Key 以及 Elasticsearch 服务是否正常运行。")

if __name__ == '__main__':
    main() 