# multimodal-embedding-retrieval

基于 DashScope 多模态向量模型与 FAISS 的本地检索示例，支持文本、图片、视频统一入库与查询。
理解多模态embedding文本、图片、视频到同一语义空间，检索--以图搜图、文搜图、图搜视频等

## 功能概览

- 多模态向量化：
  - 文本 embedding
  - 图片 embedding（Base64 Data URL）
  - 视频 embedding（本地路径字符串）
- 向量索引构建：使用 `faiss` 保存到本地
- 查询检索：
  - 相似度排序输出
  - 图片/视频意图关键词检测
  - 基于 Top-K 文本上下文生成问答结果

## 目录结构

```text
multimodal-embedding-retrieval/
├─ 1-text-embedding.py
├─ 2-image-embedding.py
├─ 3-video-embedding.py
├─ 4-build-index.py
├─ 5-query-retrieval.py
├─ requirements.txt
├─ README.md
├─ disney_knowledge_base/
│  ├─ *.docx
│  ├─ images/
│  └─ videos/
└─ index/
   ├─ disney_index.faiss
   └─ disney_metadata.json
```

## 环境准备

```bash
python -m pip install --upgrade pip
python -m pip install -r multimodal-embedding-retrieval/requirements.txt
```

## 环境变量

请先设置 DashScope Key：

```powershell
$env:DASHSCOPE_API_KEY="your_dashscope_api_key"
```

## 使用步骤

### 1) 先验证单模态 embedding（可选）

```bash
python multimodal-embedding-retrieval/1-text-embedding.py
python multimodal-embedding-retrieval/2-image-embedding.py
python multimodal-embedding-retrieval/3-video-embedding.py
```

### 2) 构建索引

```bash
python multimodal-embedding-retrieval/4-build-index.py
```

执行后会生成：

- `multimodal-embedding-retrieval/index/disney_index.faiss`
- `multimodal-embedding-retrieval/index/disney_metadata.json`

### 3) 运行查询

```bash
python multimodal-embedding-retrieval/5-query-retrieval.py
```

脚本会输出：

- 全量候选相似度排名
- 图片/视频意图识别结果
- 最终 RAG 回答（并附带匹配媒体路径）

## 关键实现说明

- 向量检索距离：`faiss.IndexFlatL2`
- 相似度换算：`similarity = 1 / (1 + distance)`
- 媒体召回策略：
  - 仅当 query 命中媒体意图关键词时触发
  - 在距离阈值内按最小距离选 Top1 媒体

## 常见问题

### 1) `faiss.read_index/write_index` 类型报错

请确保传入字符串路径（`str(path_obj)`），不要直接传 `Path` 对象。

### 2) 视频本地路径在 Windows 下报不存在

优先使用 `str(path.resolve())`，避免 `as_uri()` 在部分库里被错误解析为 `/E:/...`。

### 3) 没有检索到图片或视频

先确认：

- 索引构建时已成功处理媒体文件
- query 命中媒体关键词
- 距离阈值 `MEDIA_DISTANCE_THRESHOLD` 设置合理
