# Embedding + FAISS 检索示例

这个示例演示了如何：

- 使用 DashScope 兼容 OpenAI 接口生成文本向量
- 将向量写入 FAISS 索引
- 通过查询向量进行相似度检索
- 将索引和元数据落盘缓存，避免重复向量化，节约 token

## 文件说明

```text
embedding/
├─ embedding-faiss.py
├─ requirements.txt
└─ store/
   ├─ faiss_index.bin       # 运行后生成
   ├─ metadata_store.json   # 运行后生成
   └─ vector_ids.npy        # 运行后生成
```

## 环境依赖

安装依赖：

```bash
pip install -r embedding/requirements.txt
```

## 环境变量

脚本依赖以下环境变量：

- `DASHSCOPE_API_KEY`：DashScope API Key（必需）
- `FORCE_REEMBED`：是否强制重建文档向量（可选，默认 `false`）
- `TOP_K`：检索返回条数（可选，默认 `3`）
- `QUERY_TEXT`：查询文本（可选，默认“在线购买的门票怎么退款？”）

PowerShell 示例：

```powershell
$env:DASHSCOPE_API_KEY="你的key"
python embedding/embedding-faiss.py
```

`.env` 示例（位于 `embedding/.env`）：

```env
DASHSCOPE_API_KEY=你的key
FORCE_REEMBED=false
TOP_K=3
QUERY_TEXT=在线购买的门票怎么退款？
```

## 运行逻辑（重点）

`embedding-faiss.py` 当前策略：

1. 先检查 `store/` 目录下缓存文件是否齐全（`faiss_index.bin`、`metadata_store.json`、`vector_ids.npy`）
2. 若缓存存在且 `FORCE_REEMBED != true`：
   - 直接加载 FAISS 索引和元数据
   - 跳过文档向量化（节约 token）
3. 若缓存缺失或显式设置 `FORCE_REEMBED=true`：
   - 重新对文档生成向量
   - 重建索引并覆盖缓存文件

## 常用命令

### 1) 默认运行（优先复用缓存）

```bash
python embedding/embedding-faiss.py
```

### 2) 强制重建向量索引

PowerShell:

```powershell
$env:FORCE_REEMBED="true"
python embedding/embedding-faiss.py
```

## 输出结果

脚本会打印：

- 索引构建/加载状态
- 查询文本
- Top-K 检索结果（距离、原始文本、元数据）

## 注意事项

- 只有“文档向量化”被缓存；查询文本仍会实时生成向量
- 如果你更新了 `documents` 内容，建议设置一次 `FORCE_REEMBED=true`
- 如果需要更大规模场景，建议把元数据存储从 JSON 升级到数据库
