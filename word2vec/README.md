# 基于 Word2Vec 的中文词向量训练与相似度分析

本项目使用《****》文本进行中文分词，并基于 `gensim` 训练 `Word2Vec` 模型，演示词语相似度计算与类比查询（`most_similar`）。

## 功能概览

- 对原始中文文本进行分词（`jieba`）
- 自动处理常见文本编码（UTF-8 / GB18030 / Latin-1）
- 使用 `gensim.models.Word2Vec` 训练词向量
- 计算词语相似度（如“刘备”与“关羽”）
- 输出类比结果并保存训练模型

## 目录结构

```text
word2vec/
├─ word_seg.py
├─ word_similarity.py
├─ requirements.txt
├─ models/
│  └─ word2Vec.model
└─ three_kingdoms/
   ├─ source/
   │  └─ three_kingdoms.txt
   └─ segment/
      └─ segment_0.txt
```

## 环境要求

- 推荐 Python：`3.11` 或 `3.12`
- 已在 `requirements.txt` 中声明核心依赖：
  - `gensim==4.3.3`
  - `jieba==0.42.1`
  - `setuptools>=65,<81`

> 说明：在部分较新 Python 版本下，`gensim` 可能出现  
> `Exception ignored in: gensim.models.word2vec_inner.our_dot_float`  
> 这类底层兼容提示。推荐使用 Python 3.11/3.12 以获得更稳定体验。

## 安装依赖

在仓库根目录执行：

```bash
pip install -r word2vec/requirements.txt
```

## 使用步骤

### 1) 先做分词

```bash
python word2vec/word_seg.py
```

运行后会在 `word2vec/three_kingdoms/segment/` 生成分词文件。

### 2) 训练模型并计算相似度

```bash
python word2vec/word_similarity.py
```

运行后会：
- 输出若干相似度结果与 `most_similar` 查询结果
- 将模型保存到：`word2vec/models/word2Vec.model`

## 当前实现说明

- `word_seg.py`
  - 使用脚本所在目录构造路径，避免从不同工作目录启动时报路径错误
  - 自动创建输出目录
  - 自动尝试多种编码读取原始文本，尽可能避免 `UnicodeDecodeError`
  - 分词结果统一按 UTF-8 写出

- `word_similarity.py`
  - 使用脚本所在目录定位 `three_kingdoms/segment`
  - 训练两组参数不同的 Word2Vec 模型
  - 输出词相似度与类比查询结果
  - 自动创建 `models` 目录并保存模型文件

## 常见问题

### 1) `ValueError: input is neither a file nor a path`

通常是分词目录不存在或路径不正确。  
请先执行 `word_seg.py`，并确认 `word2vec/three_kingdoms/segment` 下有 `segment_*.txt`。

### 2) 终端中文显示乱码

这通常是控制台编码显示问题，不影响模型训练与文件写入。  
输出文件本身按 UTF-8 编码保存。

## 后续可优化方向

- 增加停用词表与自定义词典，提升分词质量
- 在训练前按句子切分与清洗，提升语义质量
- 增加评估脚本（如近义词、人名关系对比）
- 导出词向量供下游任务（文本分类、检索、推荐）使用
