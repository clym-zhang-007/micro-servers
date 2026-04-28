#!/usr/bin/env python
# coding: utf-8

"""LlamaIndex Agent 多文件 RAG 示例

功能：
- 使用 DashScope LLM + DashScope Embedding
- 加载 docs 目录下所有文件构建向量索引（支持持久化存储）
- 基于 ReAct 模式的 Agent 进行文档检索与问答
- 自动保存/加载索引，避免重复构建

用法：
- 设置环境变量 DASHSCOPE_API_KEY
- 运行 python llamaindex-agent-multi-files.py

依赖：
- pip install llama-index llama-index-llms-dashscope llama-index-embeddings-dashscope
- 环境变量 DASHSCOPE_API_KEY
"""

import os
import asyncio
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader, # LlamaIndex 的文档加载器，作用是从文件系统读取文件并转成 LlamaIndex 能处理的 Document 对象
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)


# 获取脚本所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 步骤 1：配置 LLM 和 Embedding
def setup_llm_and_embedding():
    """配置 LLM 和 Embedding，使用 DashScope"""
    api_key = os.getenv('DASHSCOPE_API_KEY')
    
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    
    # 使用 DashScope LLM
    llm = DashScope(
        model="deepseek-v3",
        api_key=api_key,
        temperature=0.7,
        top_p=0.8,
        max_tokens=4096,
    )
    
    # 使用 DashScope Embedding（自动从环境变量读取 API key）
    embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    )
    
    return llm, embed_model


# 步骤 2：加载文档并创建索引
def load_documents_and_create_index(file_dir: str = None):
    """加载文档文件夹中的所有文件并创建向量索引"""
    if file_dir is None:
        file_dir = os.path.join(BASE_DIR, 'docs')
    persist_dir = os.path.join(BASE_DIR, 'storage')
    
    if os.path.exists(persist_dir):
        try:
            # 从存储中加载索引
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print("从存储加载索引成功")
            return index
        except Exception as e:
            print(f"加载索引失败: {e}，将重新创建索引")
    
    # 如果文档目录不存在，返回空
    if not os.path.exists(file_dir):
        print(f"文档目录 {file_dir} 不存在")
        return None
    
    # 文档目录存在，且索引不存在
    # 读取文档，使用 PyMuPDF 解析 PDF 避免乱码问题
    reader = SimpleDirectoryReader(
        file_dir,
        file_extractor={
            ".pdf": PyMuPDFReader(),
        }
    )
    documents = reader.load_data()
    
    if not documents:
        print("没有找到任何文档")
        return None
    
    print(f"加载了 {len(documents)} 个文档")
    
    # 创建向量索引
    index = VectorStoreIndex.from_documents(documents)
    
    # 保存索引
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"索引已保存到 {persist_dir}")
    
    return index


# 步骤 3：创建智能体
def create_agent(index, llm):
    """创建 ReAct 智能体"""
    # 创建检索器
    retriever = index.as_retriever(similarity_top_k=5)

    # 创建查询引擎（用于检索工具）
    query_engine = index.as_query_engine(similarity_top_k=5)

    # 定义系统提示词
    system_instruction = '''你是一个专业的AI助手，请根据提供的文档内容检索信息来回答用户问题。请遵循以下要求：

        1. 严格基于检索到的文档内容作答，不要臆造或引入外部知识
        2. 若检索内容无法回答问题，请直接说明"根据现有文档，我无法回答这个问题"
        3. 回答时请注明信息来源出处（如文档标题、章节、页码等）
        4. 始终使用中文回复
    '''

    # 工具1：快速搜索（返回摘要，top_k=2）
    def quick_search(query: str) -> str:
        """快速检索，返回相关文档摘要"""
        nodes = retriever.retrieve(query)
        results = []
        for node in nodes[:2]:
            source = node.metadata.get("source", "")
            file_name = node.metadata.get("file_name", "")
            page_info = f"（第{source}页）" if source else ""
            results.append(f"[{file_name}{page_info}] {node.text[:300]}")
        return "\n\n".join(results) if results else "未找到相关内容"

    # 工具2：详细检索（返回完整内容，top_k=5）
    def detailed_search(query: str) -> str:
        """深度检索，返回完整文档片段"""
        nodes = retriever.retrieve(query)
        results = []
        for i, node in enumerate(nodes[:5]):
            source = node.metadata.get("source", "")
            file_name = node.metadata.get("file_name", "")
            page_info = f"（第{source}页）" if source else ""
            results.append(f"[{file_name}{page_info}]\n{node.text}")
        return "\n\n---\n\n".join(results) if results else "未找到相关内容"

    # 工具3：按文件名过滤检索
    def search_by_file(query: str, filename: str) -> str:
        """在指定文件中检索，需要提供文件名关键词"""
        nodes = retriever.retrieve(query)
        filtered = [n for n in nodes if filename.lower() in n.metadata.get("file_name", "").lower()]
        if not filtered:
            # 如果检索器没找到，遍历整个索引
            all_nodes = index.as_retriever().retrieve(query)
            filtered = [n for n in all_nodes if filename.lower() in n.metadata.get("file_name", "").lower()]
        results = []
        for node in filtered:
            source = node.metadata.get("source", "")
            file_name = node.metadata.get("file_name", "")
            page_info = f"（第{source}页）" if source else ""
            results.append(f"[{file_name}{page_info}]\n{node.text}")
        return "\n\n---\n\n".join(results) if results else f"未在包含 '{filename}' 的文件中找到相关内容"

    quick_tool = FunctionTool.from_defaults(fn=quick_search)
    detailed_tool = FunctionTool.from_defaults(fn=detailed_search)
    file_tool = FunctionTool.from_defaults(fn=search_by_file)

    # 创建智能体（新版 API）
    agent = ReActAgent(
        tools=[quick_tool, detailed_tool, file_tool],
        llm=llm,
        system_prompt=system_instruction,
    )

    return agent, retriever


# 步骤 4：主函数
async def main():
    """主函数"""
    # 配置 LLM 和 Embedding
    llm, embed_model = setup_llm_and_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 加载文档并创建索引
    index = load_documents_and_create_index()
    if index is None:
        print("无法创建索引，程序退出")
        return
    
    # 创建智能体
    agent, retriever = create_agent(index, llm)
    
    # 执行查询
    # query = "javascript 包含哪些数据格式"
    query = "JavaScript 在前后端交互过程中传递信息主要使用哪些数据格式"
    print(f"\n用户查询: {query}\n")
    
    # 显示召回的文档内容
    print("\n===== 召回的文档内容 =====")
    retrieved_nodes = retriever.retrieve(query)
    if retrieved_nodes:
        for i, node in enumerate(retrieved_nodes):
            print(f"\n文档片段 {i+1}:")
            source = node.metadata.get("source", "")
            file_name = node.metadata.get("file_name", "")
            page_info = f" | 第{source}页" if source else ""
            print(f"来源: {file_name}{page_info}")
            # 处理特殊字符，避免 Windows 控制台编码问题
            text_preview = node.text[:200].encode('gbk', errors='replace').decode('gbk')
            print(f"内容: {text_preview}...")  # 只显示前200个字符
            if hasattr(node, 'score'):
                print(f"相似度分数: {node.score}")
    else:
        print("没有召回任何文档内容")
    print("===========================\n")
    
    # 使用智能体回答问题（新版 API 使用 run 方法，是异步的）
    print("\n===== 智能体回复 =====")
    response = await agent.run(query)
    # 处理特殊字符，避免 Windows 控制台编码问题
    response_str = str(response).encode('gbk', errors='replace').decode('gbk')
    print(response_str)
    print("======================\n")


if __name__ == "__main__":
    asyncio.run(main())
    