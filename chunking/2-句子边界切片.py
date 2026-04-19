#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
句子边界切片策略
基于句子边界进行切分，保持语义完整性
"""

import re

# 仅含句末标点的片段（用于 split 后识别分隔符）
_SENTENCE_PUNCT_ONLY = re.compile(r"^[.!?。！？…]+$")

# 超长句子的次级切分：优先在分号、逗号、顿号处断开，最后再硬切
_SOFT_BREAK_RE = re.compile(r"[；;，,、]")


def _split_paragraph_into_sentences(para: str) -> list[str]:
    """按句末标点拆句，并保留标点（re.split 默认会丢掉分隔符）。"""
    parts = re.split(r"([.!?。！？…]+)", para)
    sentences: list[str] = []
    buf = ""
    for p in parts:
        if p == "":
            continue
        if _SENTENCE_PUNCT_ONLY.match(p):
            buf += p
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
        else:
            buf += p
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


def _split_text_into_sentences(text: str) -> list[str]:
    """按换行分段，再逐段拆句并保留句末标点。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        out.extend(_split_paragraph_into_sentences(para))
    return out


def _join_fragments(prefix: str, addition: str) -> str:
    """中英文混排时：英文词之间用空格；中文与中文之间一般不插空格。"""
    if not prefix:
        return addition
    # 两侧均为 ASCII 字母/数字时加空格，避免英文粘连
    if prefix[-1].isascii() and addition[0].isascii():
        return prefix + " " + addition
    return prefix + addition


def _split_oversized(text: str, max_chunk_size: int) -> list[str]:
    """单句超过 max_chunk_size 时：在窗口内优先找最后一个软标点断句，否则按固定长度硬切。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chunk_size:
        return [text]

    out: list[str] = []
    start = 0
    while start < len(text):
        remain = len(text) - start
        if remain <= max_chunk_size:
            out.append(text[start:].strip())
            break

        window_end = min(start + max_chunk_size, len(text))
        window = text[start:window_end]
        best_pos = -1  # 在原文中的切分终点（不含该字符之后的内容）
        for m in _SOFT_BREAK_RE.finditer(window):
            best_pos = start + m.end()

        min_piece = max(1, max_chunk_size // 4)
        if best_pos > start + min_piece:
            out.append(text[start:best_pos].strip())
            start = best_pos
        else:
            out.append(text[start : start + max_chunk_size].strip())
            start += max_chunk_size

    return [p for p in out if p]


def semantic_chunking(text, max_chunk_size=512):
    """基于句子边界的切片：先拆句，再按 max_chunk_size 合并；超长单句会二次切分。

    说明：
    - 小数点、英文缩写（如 U.S.A.）可能被误切，纯英文场景建议换用 spaCy 等分句。
    - 合并时尽量避免在中文句间插入英文空格。
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size 必须为正整数")

    sentences = _split_text_into_sentences(text)
    # 展开超长句，避免单个 chunk 超过上限
    expanded: list[str] = []
    for sentence in sentences:
        expanded.extend(_split_oversized(sentence, max_chunk_size))

    chunks: list[str] = []
    current_chunk = ""

    for sentence in expanded:
        candidate = _join_fragments(current_chunk, sentence) if current_chunk else sentence
        if len(candidate) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        elif len(candidate) > max_chunk_size and not current_chunk:
            # 理论上 _split_oversized 已处理；兜底再切
            chunks.extend(_split_oversized(sentence, max_chunk_size))
            current_chunk = ""
        else:
            current_chunk = candidate

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def print_chunk_analysis(chunks, method_name):
    """打印切片分析结果"""
    print(f"\n{'='*60}")
    print(f"📋 {method_name}")
    print(f"{'='*60}")
    
    if not chunks:
        print("❌ 未生成任何切片")
        return
    
    total_length = sum(len(chunk) for chunk in chunks)
    avg_length = total_length / len(chunks)
    min_length = min(len(chunk) for chunk in chunks)
    max_length = max(len(chunk) for chunk in chunks)
    
    print(f"📊 统计信息:")
    print(f"   - 切片数量: {len(chunks)}")
    print(f"   - 平均长度: {avg_length:.1f} 字符")
    print(f"   - 最短长度: {min_length} 字符")
    print(f"   - 最长长度: {max_length} 字符")
    print(f"   - 长度方差: {max_length - min_length} 字符")
    
    print(f"\n📝 切片内容:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   块 {i} ({len(chunk)} 字符):")
        print(f"   {chunk}")
        print()

# 测试文本
text = """
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

if __name__ == "__main__":
    print("🎯 句子边界切片策略测试")
    print(f"📄 测试文本长度: {len(text)} 字符")
    
    # 使用句子边界切片
    chunks = semantic_chunking(text, max_chunk_size=200)
    print_chunk_analysis(chunks, "句子边界切片")