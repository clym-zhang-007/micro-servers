#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片策略对比测试脚本
从同目录 1～6 号 Python 模块加载切片函数，统一对比展示。
"""

import importlib.util
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def _load_chunk_module(filename: str):
    """按文件路径加载模块（文件名含中文，不宜直接 import）。"""
    path = _DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"未找到切片模块: {path}")
    mod_name = "chunking_" + filename.replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建模块 spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 与 1-固定长度切片.py … 6-自适应切片.py 一一对应
_m1 = _load_chunk_module("1-固定长度切片.py")
_m2 = _load_chunk_module("2-句子边界切片.py")
_m3 = _load_chunk_module("3-LLM语义切片.py")
_m4 = _load_chunk_module("4-层次切片.py")
_m5 = _load_chunk_module("5-滑动窗口切片.py")
_m6 = _load_chunk_module("6-自适应切片.py")

improved_fixed_length_chunking = _m1.improved_fixed_length_chunking
semantic_chunking = _m2.semantic_chunking
advanced_semantic_chunking_with_llm = _m3.advanced_semantic_chunking_with_llm
hierarchical_chunking = _m4.hierarchical_chunking
sliding_window_chunking = _m5.sliding_window_chunking
adaptive_chunking = _m6.adaptive_chunking


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


def main():
    """主测试函数"""
    text = """
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

    print("🎯 切片策略对比测试")
    print(f"📄 测试文本长度: {len(text)} 字符")

    target_size = 300

    chunks1 = improved_fixed_length_chunking(text, chunk_size=target_size, overlap=50)
    print_chunk_analysis(chunks1, "1. 固定长度切片（1-固定长度切片.py）")

    chunks2 = semantic_chunking(text, max_chunk_size=target_size)
    print_chunk_analysis(chunks2, "2. 句子边界切片（2-句子边界切片.py）")

    print("\n🤖 正在调用 LLM 进行语义切片（3-LLM语义切片.py）...")
    chunks3 = advanced_semantic_chunking_with_llm(text, max_chunk_size=target_size)
    print_chunk_analysis(chunks3, "3. LLM语义切片（3-LLM语义切片.py）")

    chunks4 = hierarchical_chunking(text, target_size=target_size, preserve_hierarchy=True)
    print_chunk_analysis(chunks4, "4. 层次切片（4-层次切片.py）")

    chunks5 = sliding_window_chunking(text, window_size=target_size, step_size=target_size // 2)
    print_chunk_analysis(chunks5, "5. 滑动窗口切片（5-滑动窗口切片.py）")

    chunks6 = adaptive_chunking(text, target_size=target_size, tolerance=0.3)
    print_chunk_analysis(chunks6, "6. 自适应切片（6-自适应切片.py）")

    print(f"\n{'='*80}")
    print("📈 策略对比总结")
    print(f"{'='*80}")

    methods = [
        ("固定长度", chunks1),
        ("句子边界", chunks2),
        ("LLM语义", chunks3),
        ("层次切片", chunks4),
        ("滑动窗口", chunks5),
        ("自适应", chunks6),
    ]

    print(f"{'策略':<12} {'切片数':<6} {'平均长度':<8} {'长度方差':<8} {'推荐度':<8}")
    print("-" * 50)

    for name, chunks in methods:
        if chunks:
            avg_len = sum(len(c) for c in chunks) / len(chunks)
            min_len = min(len(c) for c in chunks)
            max_len = max(len(c) for c in chunks)
            variance = max_len - min_len

            if len(chunks) >= 2 and variance < 100 and avg_len > 150:
                recommendation = "⭐⭐⭐⭐⭐"
            elif len(chunks) >= 2 and variance < 150:
                recommendation = "⭐⭐⭐⭐"
            elif len(chunks) >= 1:
                recommendation = "⭐⭐⭐"
            else:
                recommendation = "⭐⭐"

            print(f"{name:<12} {len(chunks):<6} {avg_len:<8.1f} {variance:<8.1f} {recommendation:<8}")
        else:
            print(f"{name:<12} {'0':<6} {'N/A':<8} {'N/A':<8} {'⭐':<8}")


if __name__ == "__main__":
    main()
