# Agent Framework Compare

基于不同 Agent 框架实现的多文件 RAG（检索增强生成）问答应用示例。

## 项目结构

```
agent-framework-compare/
├── langchain-agent-multi-files.py    # 基于 LangChain 的 RAG 实现
├── llamaindex-agent-multi-files.py   # 基于 LlamaIndex 的 ReAct Agent 实现
├── qwen-agent-multi-files.py         # 基于 Qwen Agent 的多工具 Agent 实现
├── docs/                             # 文档目录（PDF / TXT）
├── requirements.txt                  # Python 依赖
├── storage/                          # LlamaIndex 向量索引持久化目录
└── langchain_storage/                # LangChain FAISS 索引持久化目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 API Key

```bash
# Linux / macOS
export DASHSCOPE_API_KEY="your-api-key"

# Windows (CMD)
set DASHSCOPE_API_KEY=your-api-key

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="your-api-key"
```

### 3. 准备文档

将文档文件（`.txt` / `.pdf`）放入 `docs/` 目录。

### 4. 运行

```bash
# LangChain 示例
python langchain-agent-multi-files.py

# LlamaIndex 示例
python llamaindex-agent-multi-files.py

# Qwen Agent 示例（默认启动 Web UI）
python qwen-agent-multi-files.py
```

## 各框架对比

| 特性 | LangChain | LlamaIndex | Qwen Agent |
|------|-----------|------------|------------|
| **核心模式** | LCEL 管道链式调用 | ReAct Agent 自主规划 | Assistant 内置工具 |
| **向量库** | FAISS | 内置 VectorStoreIndex | 内置 |
| **多工具支持** | 需手动组装 Chain | 原生支持，Agent 自主选择 | 原生支持 |
| **PDF 解析** | 仅 TXT（可扩展） | PyMuPDF（按页拆分） | 框架内置 |
| **持久化** | FAISS local | StorageContext persist | 无 |
| **适合场景** | 快速构建标准化 RAG 流水线 | 复杂检索 + Agent 决策 | 快速原型 + 多模态工具 |

## 依赖说明

- **LangChain 相关**: `langchain-community`, `langchain-core`, `langchain-text-splitters`, `faiss-cpu`
- **LlamaIndex 相关**: `llama-index-core`, `llama-index-llms-dashscope`, `llama-index-embeddings-dashscope`, `llama-index-readers-file`
- **PDF 解析**: `PyMuPDF`
- **DashScope SDK**: `dashscope`
- **Qwen Agent**: `qwen-agent`

所有示例均使用 DashScope（通义千问）作为 LLM 和 Embedding 服务。

## 注意事项

- 首次运行会自动构建向量索引，后续运行会从持久化存储加载
- 如需重新构建索引，删除 `storage/` 或 `langchain_storage/` 目录即可
- PDF 解析依赖 `PyMuPDF`，确保已安装
