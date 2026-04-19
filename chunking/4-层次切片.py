#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
层次切片策略
按 Markdown 标题切分为「小节」，小节内再按目标长度用句界打包，避免在标题/小节中间随意切断。
"""

import re

# 句末标点保留（与 2-句子边界切片 一致思路）
_SENTENCE_PUNCT_ONLY = re.compile(r"^[.!?。！？…]+$")
_SOFT_BREAK_RE = re.compile(r"[；;，,、]")

# 识别 Markdown 标题行：行首 # 且 # 后有空格类分隔
_HEADING_LINE = re.compile(r"^#+\s+\S")


def _split_paragraph_into_sentences(para: str) -> list[str]:
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


def _split_oversized(text: str, max_chunk_size: int) -> list[str]:
    """单段仍超长时按软标点/硬切（与句子边界脚本同思路）。"""
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
        best_pos = -1
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


def _split_into_md_sections(text: str) -> list[str]:
    """按标题行切分：每个 # / ## / ### … 开启新的小节，避免多个 ### 混在同一块。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    sections: list[str] = []
    buf: list[str] = []

    for line in lines:
        s = line.strip()
        if s and _HEADING_LINE.match(s):
            if buf:
                block = "\n".join(buf).strip()
                if block:
                    sections.append(block)
            buf = [line]
        else:
            buf.append(line)

    if buf:
        block = "\n".join(buf).strip()
        if block:
            sections.append(block)
    return sections


def _pack_section(section: str, target_size: int) -> list[str]:
    """
    单个小节若超长：第一行作为小节标题；正文按句拆开后打包；
    每个输出块都重复小节标题，便于向量检索带上下文。
    """
    lines = section.split("\n")
    head = lines[0].strip()
    body = "\n".join(lines[1:]).strip()

    if not body:
        return [head] if head else []

    full = f"{head}\n\n{body}"
    if len(full) <= target_size:
        return [full]

    sentences: list[str] = []
    for para in body.split("\n"):
        p = para.strip()
        if p:
            sentences.extend(_split_paragraph_into_sentences(p))
    if not sentences:
        return _split_oversized(full, target_size)

    header_prefix = head + "\n\n"
    overhead = len(header_prefix)
    chunks: list[str] = []
    acc: list[str] = []

    def flush_acc() -> None:
        if acc:
            chunks.append(header_prefix + "\n".join(acc))
            acc.clear()

    for sent in sentences:
        if overhead + len(sent) > target_size:
            flush_acc()
            for piece in _split_oversized(sent, max(1, target_size - overhead)):
                chunks.append(header_prefix + piece)
            continue

        candidate = "\n".join(acc + [sent]) if acc else sent
        if overhead + len(candidate) > target_size:
            flush_acc()
            acc.append(sent)
        else:
            acc.append(sent)

    flush_acc()
    return chunks


def hierarchical_chunking(text, target_size=512, preserve_hierarchy=True):
    """层次切片：先按标题拆成小节，再在小节内按句与 target_size 打包。

    - 每个 Markdown 标题行开启新小节，### 不再与上一小节混写。
    - 小节超长时在句界切开，并重复小节标题（preserve_hierarchy=True 时生效）。
    - 不再使用「段落行 + 80% 长度」这类易误切的启发式。
    """
    if target_size <= 0:
        raise ValueError("target_size 必须为正整数")

    text = text.strip()
    if not text:
        return []

    sections = _split_into_md_sections(text)
    if not sections:
        return [text]

    all_chunks: list[str] = []
    for sec in sections:
        parts = _pack_section(sec, target_size)
        # preserve_hierarchy：小节内续块默认仍带同一标题；若仅需「标题只在首块出现」可在此对 parts[1:] 去标题
        if not preserve_hierarchy and len(parts) > 1:
            head_line = sec.split("\n", 1)[0].strip()
            merged = [parts[0]]
            prefix = head_line + "\n\n"
            for p in parts[1:]:
                if p.startswith(prefix):
                    merged.append(p[len(prefix) :].lstrip())
                else:
                    merged.append(p)
            all_chunks.extend(merged)
        else:
            all_chunks.extend(parts)

    return all_chunks


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


# 测试文本 - 包含层次结构
text = """
# 迪士尼乐园门票指南

## 一、门票类型介绍

### 1. 基础门票类型
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

### 2. 特殊门票类型
年票适合经常游玩的游客，提供更多优惠和特权。VIP门票包含快速通道服务，可减少排队时间。团体票适用于10人以上团队，享受团体折扣。

## 二、购票渠道与流程

### 1. 官方购票渠道
购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。这些渠道提供最可靠的服务和最新的票务信息。

### 2. 第三方平台
第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。建议优先选择官方渠道以确保购票安全。

### 3. 证件要求
所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

## 三、入园须知

### 1. 入园时间
乐园通常在上午8:00开园，晚上8:00闭园，具体时间可能因季节和特殊活动调整。建议提前30分钟到达园区。

### 2. 安全检查
入园前需要进行安全检查，禁止携带危险物品、玻璃制品等。建议轻装简行，提高入园效率。

### 3. 园区服务
园区内提供寄存服务、轮椅租赁、婴儿车租赁等服务，可在游客服务中心咨询详情。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

if __name__ == "__main__":
    print("🎯 层次切片策略测试")
    print(f"📄 测试文本长度: {len(text)} 字符")

    chunks = hierarchical_chunking(text, target_size=200, preserve_hierarchy=True)
    print_chunk_analysis(chunks, "层次切片")
