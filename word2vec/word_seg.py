# -*-coding: utf-8 -*-
# 对txt文件进行中文分词
import jieba
import os
from pathlib import Path

# 基于脚本所在目录构建路径，避免从不同 cwd 启动时路径失效
BASE_DIR = Path(__file__).resolve().parent
# 源文件所在目录
source_folder = str(BASE_DIR / 'three_kingdoms' / 'source')
segment_folder = str(BASE_DIR / 'three_kingdoms' / 'segment')

# 字词分割，对整个文件内容进行字词分割
def segment_lines(file_list, segment_out_dir, stopwords=None):
    # 避免使用可变默认参数，防止函数多次调用时状态污染
    stopwords = set(stopwords or [])
    os.makedirs(segment_out_dir, exist_ok=True)
    for i, file in enumerate(file_list):
        segment_out_name = os.path.join(segment_out_dir, 'segment_{}.txt'.format(i))
        # 自动兼容常见中文文本编码，避免系统默认编码（如 gbk）导致解码失败
        with open(file, 'rb') as f:
            raw = f.read()
        document = None
        for enc in ('utf-8', 'gb18030', 'latin-1'):
            try:
                document = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if document is None:
            # 兜底：替换非法字符，避免任务中断
            document = raw.decode('utf-8', errors='replace')

        document_cut = jieba.cut(document)
        sentence_segment = []
        for word in document_cut:
            if word not in stopwords:
                sentence_segment.append(word)
        result = ' '.join(sentence_segment)
        # 输出统一为 UTF-8，便于后续训练/分析流程复用
        with open(segment_out_name, 'w', encoding='utf-8') as f2:
            f2.write(result)

# 对 source 中的 txt 文件进行分词，输出到 segment 目录中
file_list = [str(p) for p in sorted(Path(source_folder).rglob("*.txt"))]
segment_lines(file_list, segment_folder)
print("分词完成，处理文件数：", len(file_list))
print("输出目录：", segment_folder)
