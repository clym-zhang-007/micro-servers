#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
固定长度切片策略
在句子边界进行切分，避免切断句子
"""

def improved_fixed_length_chunking(text, chunk_size=512, overlap=50):
    """改进的固定长度切片：目标长度为 chunk_size，尽量在句末/分号处截断，块间 overlap 重叠。

    注意：
    - 必须满足 overlap < chunk_size，否则无法向前推进，会死循环。
    - 对块做 strip() 仅影响存入内容；下一块的起始仍按未 strip 的 end/overlap 计算，
      若块首尾有大量空白，重叠语义会与「纯文本字符」略有偏差，属可接受折中。
    - 英文小数（如 3.14）、缩写（Mr.）可能被 `.` 误切，纯英文场景可改用更细规则或 NLP 分句。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if overlap < 0:
        raise ValueError("overlap 不能为负")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size，否则切片无法向前移动")

    chunks = []
    start = 0
    # 从目标截断点向前回溯，搜索句末标点的窗口（与 chunk_size 成比例，避免大块时搜不到句号）
    sentence_lookback = min(chunk_size, max(80, chunk_size // 3))
    sentence_end_chars = frozenset('.!?。！？；;')

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # 非最后一段时，尽量在窗口内从后往前找最近断句点，避免在句中硬切
        if end < len(text):
            search_from = end - 1
            search_to = max(start, end - sentence_lookback) - 1
            for i in range(search_from, search_to, -1):
                if text[i] in sentence_end_chars:
                    end = i + 1
                    break

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break
        start = end - overlap

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
    print("🎯 固定长度切片策略测试")
    print(f"📄 测试文本长度: {len(text)} 字符")
    
    # 使用改进的固定长度切片
    chunks = improved_fixed_length_chunking(text, chunk_size=200, overlap=20)
    print_chunk_analysis(chunks, "固定长度切片")