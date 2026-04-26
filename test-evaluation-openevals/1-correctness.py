#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OpenEvals 正确性评估器示例

这个脚本演示如何使用 OpenEvals 的 LLM-as-Judge 评估器，对模型输出进行 correctness（正确性）评分。

核心流程：
1. 从环境变量读取 DashScope API Key。
2. 使用 LangChain 的 ChatTongyi 创建一个评估用 LLM。
3. 使用 OpenEvals 内置的 CORRECTNESS_PROMPT 创建正确性评估器。
4. 分别对单个样例和多个不同质量样例进行评估。
5. 打印评估结果结构，帮助理解 OpenEvals evaluator 的返回格式。

注意：
- 这里的 LLM 不是被评估的业务模型，而是“裁判模型”。
- evaluator 会把 input、output、reference_output 交给裁判模型判断 output 是否符合参考答案。
- correctness 分数通常在 0.0 到 1.0 之间，越高表示越正确。
"""

import os
import traceback

from langchain_community.chat_models import ChatTongyi
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

# ==================== 配置 ====================

# 从系统环境变量中读取 DashScope API Key。
# ChatTongyi 底层会调用阿里云 DashScope 的通义千问模型，因此必须提供有效密钥。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 如果没有配置 API Key，脚本无法创建评估模型，因此直接给出配置提示并退出。
if not DASHSCOPE_API_KEY:
    print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
    print("Windows PowerShell: $env:DASHSCOPE_API_KEY='your-key'")
    print("Linux/Mac: export DASHSCOPE_API_KEY='your-key'")
    exit(1)

# ==================== 创建评估 LLM ====================

print("=" * 60)
print("创建评估 LLM...")
print("=" * 60)

# 创建一个专门用于“打分/裁判”的 LLM。
# model_name="qwen-turbo" 表示使用通义千问 turbo 模型作为评估模型。
# temperature=0 表示尽量降低随机性，让同一组输入的评估结果更稳定。
eval_llm = ChatTongyi(
    model_name="qwen-turbo",
    dashscope_api_key=DASHSCOPE_API_KEY,
    temperature=0,
)

print("[OK] 评估 LLM 创建成功\n")

# ==================== 创建评估器 ====================

print("=" * 60)
print("创建正确性评估器...")
print("=" * 60)

# create_llm_as_judge 会把一个 LLM 包装成 evaluator。
# 这个 evaluator 接收 inputs、outputs、reference_outputs，然后让裁判模型根据 prompt 给 outputs 打分。
evaluator = create_llm_as_judge(
    # OpenEvals 内置的正确性评估提示词，用于判断输出是否符合参考答案。
    prompt=CORRECTNESS_PROMPT,
    # feedback_key 会出现在评估结果中，用来标识这个分数对应的评估维度。
    # 在返回的结果中用这个字段作为key的值
    feedback_key="correctness",
    # judge 指定真正执行判断的裁判模型。
    judge=eval_llm,
    # continuous=True 表示输出连续分数，例如 0.0、0.6、0.8、1.0。
    # 如果设为 False，评估结果可能更偏向二分类或离散判断。
    continuous=True,
    # use_reasoning=False 表示只要求裁判模型返回分数，不额外返回详细推理过程。
    # 如果希望看到模型为什么这样打分，可以改成 True。
    use_reasoning=False,
)

print("[OK] 评估器创建成功\n")

# ==================== 测试评估 ====================

print("=" * 60)
print("开始评估测试...")
print("=" * 60)

# 测试用例 1：基本评估。
# inputs 是用户原始问题，outputs 是待评估回答，reference_outputs 是标准/参考答案。
print("\n【测试用例 1】")
print("-" * 60)
inputs_1 = "什么是机器学习？"
outputs_1 = "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。"
reference_outputs_1 = "机器学习是让计算机从数据中学习的技术。"

print(f"输入: {inputs_1}")
print(f"输出: {outputs_1}")
print(f"参考输出: {reference_outputs_1}")
print("\n正在评估...")

try:
    # 调用 evaluator 执行正确性评估。
    # OpenEvals 会把这三个字段组合进 CORRECTNESS_PROMPT，再调用 eval_llm 生成评分。
    result_1 = evaluator(
        inputs=inputs_1,
        outputs=outputs_1,
        reference_outputs=reference_outputs_1,
    )

    print("\n[OK] 评估完成")
    print(f"\n评估结果类型: {type(result_1)}")
    print(f"评估结果内容:\n{result_1}")

    # OpenEvals 的不同配置或版本可能返回不同结构。
    # 这里用多分支判断返回类型，方便观察和兼容各种返回格式。
    if isinstance(result_1, dict):
        # 常见返回格式：{'key': 'correctness', 'score': 0.8, 'comment': None, 'metadata': None}
        print(f"\n结果字典键: {result_1.keys()}")
        if "score" in result_1:
            print(f"分数: {result_1['score']}")
        if "key" in result_1:
            print(f"评估键: {result_1['key']}")
        if "comment" in result_1:
            print(f"评论: {result_1['comment']}")
    elif isinstance(result_1, (int, float)):
        # 某些 evaluator 可能直接返回一个数值分数。
        print(f"\n分数: {result_1}")
    elif isinstance(result_1, tuple):
        # 当 use_reasoning=True 时，返回值可能包含分数和理由。
        print(f"\n元组内容: {result_1}")
        if len(result_1) >= 1:
            print(f"分数: {result_1[0]}")
        if len(result_1) >= 2:
            print(f"理由: {result_1[1]}")

except Exception as e:
    # 如果评估失败，通常可能是 API Key、网络、模型调用或 evaluator 参数格式问题。
    print(f"\n❌ 评估失败: {e}")
    traceback.print_exc()

# 测试用例 2：不同质量的输出。
# 通过高质量、部分正确、错误回答三个样例，观察 correctness 分数如何变化。
print("\n\n" + "=" * 60)
print("【测试用例 2】- 测试不同质量的输出")
print("=" * 60)

test_cases = [
    {
        "name": "高质量回答",
        "inputs": "什么是Python？",
        "outputs": "Python是一种高级编程语言，具有简洁的语法和强大的功能。",
        "reference_outputs": "Python是一种高级编程语言。",
    },
    {
        "name": "部分正确回答",
        "inputs": "什么是Python？",
        "outputs": "Python是一种编程语言。",
        "reference_outputs": "Python是一种高级编程语言，具有简洁的语法和强大的功能。",
    },
    {
        "name": "错误回答",
        "inputs": "什么是Python？",
        "outputs": "Python是一种数据库管理系统。",
        "reference_outputs": "Python是一种高级编程语言。",
    },
]

# 逐条运行测试用例，分别输出每条样例的评估分数。
for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- 测试 {i}: {test_case['name']} ---")
    print(f"输入: {test_case['inputs']}")
    print(f"输出: {test_case['outputs']}")
    print(f"参考输出: {test_case['reference_outputs']}")

    try:
        # 使用同一个 evaluator 评估当前测试样例。
        result = evaluator(
            inputs=test_case["inputs"],
            outputs=test_case["outputs"],
            reference_outputs=test_case["reference_outputs"],
        )

        # 从返回结果中提取 score。
        # 这里仍然兼容 dict、数字、tuple 三种可能的返回结构。
        score = None
        if isinstance(result, dict):
            score = result.get("score")
        elif isinstance(result, (int, float)):
            score = result
        elif isinstance(result, tuple) and len(result) > 0:
            score = result[0]

        # 输出格式化分数，保留三位小数，便于对比不同测试样例。
        if score is not None:
            print(f"[OK] 评估分数: {score:.3f}")
        else:
            print(f"[WARN] 无法提取分数，结果: {result}")

    except Exception as e:
        # 单条样例失败时只打印错误，不影响后续样例继续执行。
        print(f"[ERROR] 评估失败: {e}")

# ==================== 总结 ====================

print("\n\n" + "=" * 60)
print("测试完成")
print("=" * 60)
print("\n说明:")
print("1. 评估结果格式: 字典 {'key': 'correctness', 'score': 0.8, 'comment': None, 'metadata': None}")
print("2. 分数范围: 0.0 到 1.0")
print("3. 分数含义: 分数越高表示输出越正确")
print("   - 0.8-1.0: 高质量回答")
print("   - 0.5-0.8: 部分正确回答")
print("   - 0.0-0.5: 低质量或错误回答")
print("4. 如果 use_reasoning=True，结果可能是 (分数, 理由) 元组")
print("5. 如果 use_reasoning=False，结果通常是包含分数的字典")
print("\n实际运行结果示例:")
print("  - 高质量回答: score = 0.800")
print("  - 部分正确回答: score = 0.600")
print("  - 错误回答: score = 0.000")
