#!/usr/bin/env python
# coding: utf-8

# 使用BAAI/bge-reranker-large模型进行rerank
# 使用AutoTokenizer和AutoModelForSequenceClassification加载模型
# reranker是将问题与文档一起输入进行交叉熵计算，得到相关性分数

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-reranker-large', cache_dir='/root/autodl-tmp/models')

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('/root/autodl-tmp/models/BAAI/bge-reranker-large')
# 设置模型为评估模式
model.eval()

pairs = [['what is panda?', 'The giant panda is a bear species endemic to China.']]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
scores = model(**inputs).logits.view(-1).float()
print(scores)  # 输出相关性分数

pairs = [
    ['what is panda?', 'The giant panda is a bear species endemic to China.'],  # 高相关
    ['what is panda?', 'Pandas are cute.'],                                     # 中等相关
    ['what is panda?', 'The Eiffel Tower is in Paris.']                        # 不相关
]
# pairs: 输入文本对列表
# padding=True: 对较短的序列进行填充，使批次中所有序列长度一致
# truncation=True: 截断超过max_length的序列
# return_tensors='pt': 返回PyTorch张量
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
# .logits: 模型输出层（分类器）的原始未归一化分数，范围通常为[-inf, inf]
# view(-1): 将多维张量展平为一维，便于后续处理
# float(): 将分数转换为浮点数，便于后续计算
scores = model(**inputs).logits.view(-1).float()
print(scores)  # 输出相关性分数

