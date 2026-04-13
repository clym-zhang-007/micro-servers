# -*-coding: utf-8 -*-
# 先运行 word_seg进行中文分词，然后再进行word_similarity计算
# 将Word转换成Vec，然后计算相似度 
from gensim.models import word2vec
import multiprocessing
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 如果目录中有多个文件，可以使用PathLineSentences
segment_folder = str(BASE_DIR / 'three_kingdoms' / 'segment')
# 切分之后的句子合集
sentences = word2vec.PathLineSentences(segment_folder)

# 设置模型参数，进行训练
model = word2vec.Word2Vec(sentences, vector_size=128, window=3, min_count=1)
# print(model.wv['刘备'])
print(model.wv.similarity('刘备', '关羽'))
print(model.wv.similarity('刘备', '张飞'))
print(model.wv.most_similar(positive=['刘备', '关羽'], negative=['张飞']))
print('--------------------------------')
# 设置模型参数，进行训练
model2 = word2vec.Word2Vec(sentences, vector_size=512, window=5, min_count=2, workers=multiprocessing.cpu_count())
print(model2.wv.similarity('刘备', '关羽'))
print(model2.wv.similarity('刘备', '张飞'))
print(model2.wv.most_similar(positive=['刘备', '关羽'], negative=['张飞']))
print('--------------------------------')
# 保存模型
models_dir = BASE_DIR / 'models'
models_dir.mkdir(parents=True, exist_ok=True)
model2.save(str(models_dir / 'word2Vec.model'))
