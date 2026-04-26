#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepEval 评估脚本 - 投顾AI助手质量测试

使用 DeepEval 对投顾AI助手进行自动化质量评估，包括：
1. 答案相关性评估 - 回答是否切题
2. 幻觉检测 - 是否编造虚假信息
3. 自定义评估 - 是否考虑客户风险偏好

安装依赖：
    pip install deepeval

运行方式：
    方式1: python deepeval_wealth_advisor.py
    方式2: deepeval test run deepeval_wealth_advisor.py

环境变量（DeepEval 需要 OpenAI 作为评估器）：
    OPENAI_API_KEY: OpenAI API 密钥
    DASHSCOPE_API_KEY: 通义千问 API 密钥（运行投顾助手）
"""

import os
from datetime import datetime
from typing import Dict, Any, List

# DeepEval 是一个专门用于评估 LLM 应用输出质量的框架。
# evaluate：批量运行测试用例和评估指标。
# LLMTestCase：DeepEval 的标准测试用例结构，用来描述输入、实际输出、上下文等。
# LLMTestCaseParams：在 GEval 自定义指标中指定要让评估模型参考哪些字段。
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# AnswerRelevancyMetric：评估回答是否与用户问题相关。
# HallucinationMetric：评估回答是否基于上下文，是否存在幻觉。
# GEval：使用 LLM 作为裁判，根据自定义 criteria 和 steps 进行评分。
from deepeval.metrics import (
    AnswerRelevancyMetric,
    HallucinationMetric,
    GEval
)

# 下面开始动态导入投顾助手主程序。
# 由于主程序文件名是 1-hybrid_wealth_advisor_langgraph_langsmith.py，
# 文件名以数字开头，不能直接使用普通 import 语法导入，必须使用 importlib 按文件路径加载。
import importlib.util
import sys

# 创建模块加载规范。
# 第一个参数是给目标文件起的临时模块名。
# 第二个参数是实际要加载的 Python 文件路径。
spec = importlib.util.spec_from_file_location(
    "hybrid_wealth_advisor_langgraph_langsmith",
    "1-hybrid_wealth_advisor_langgraph_langsmith.py"
)

# 根据 spec 创建模块对象。此时只是创建对象，目标文件中的代码还没有执行。
hybrid_module = importlib.util.module_from_spec(spec)

# 把动态模块注册到 sys.modules。
# 这样后续 from hybrid_wealth_advisor_langgraph_langsmith import ... 才能正常工作。
sys.modules["hybrid_wealth_advisor_langgraph_langsmith"] = hybrid_module

# 执行目标文件代码，完成模块初始化。
spec.loader.exec_module(hybrid_module)

# run_wealth_advisor：真实的投顾助手入口函数，评估时会反复调用它生成回答。
# SAMPLE_CUSTOMER_PROFILES：示例客户画像，用于补充测试上下文和风险偏好信息。
from hybrid_wealth_advisor_langgraph_langsmith import (
    run_wealth_advisor,
    SAMPLE_CUSTOMER_PROFILES
)


# ==================== 测试用例定义 ====================

# 反应式查询测试用例，也就是简单问答类问题。
# 这类问题通常不需要复杂规划，只需要智能体快速给出解释或查询结果。
# 每个测试数据包含：
# - input：用户问题。
# - customer_id：使用哪个模拟客户画像。
# - context：DeepEval 幻觉检测会参考的背景上下文。
# - expected_keywords：预期关键词，目前本脚本没有直接用于 DeepEval 指标，只是保留为人工检查或后续扩展使用。
REACTIVE_TEST_CASES = [
    {
        "input": "今天上证指数表现如何？",
        "customer_id": "customer1",
        "context": ["上证指数是中国股市的重要指标", "提供实时行情数据"],
        "expected_keywords": ["上证指数", "点位"]
    },
    {
        "input": "请解释一下什么是ETF？",
        "customer_id": "customer1",
        "context": ["ETF是交易所交易基金", "可以在交易所买卖"],
        "expected_keywords": ["ETF", "基金", "交易"]
    },
]

# 深思熟虑查询测试用例，也就是复杂投资分析、规划、建议类问题。
# 这类问题更关注回答是否结合客户画像、风险偏好、投资期限等因素。
DELIBERATIVE_TEST_CASES = [
    {
        "input": "根据我的风险偏好，应该如何调整投资组合？",
        "customer_id": "customer1",
        "context": [
            "客户风险评级：平衡型",
            "当前配置：股票40%，债券30%，现金10%，另类投资20%",
            "投资期限：中期"
        ],
        "expected_keywords": ["风险", "配置", "建议"]
    },
    {
        "input": "请帮我制定一个退休投资计划",
        "customer_id": "customer2",
        "context": [
            "客户风险评级：进取型",
            "财务目标：财富增长、资产配置多元化",
            "投资期限：长期"
        ],
        "expected_keywords": ["退休", "规划", "投资"]
    },
]


# ==================== 辅助函数 ====================

def run_agent_and_get_response(query: str, customer_id: str) -> Dict[str, Any]:
    """
    运行投顾助手并获取响应。

    参数：
    - query：用户输入的问题。
    - customer_id：模拟客户 ID，用于让投顾助手加载对应客户画像。

    返回：
    - run_wealth_advisor 的完整返回结果，通常包含 final_response、processing_mode、query_type 等字段。
    """
    # 打印当前测试输入，方便在命令行中观察评估进度。
    print(f"\n[测试] 运行查询: {query[:50]}...")

    # 调用真实投顾助手。DeepEval 评估的是这个函数生成的实际回答质量。
    result = run_wealth_advisor(user_query=query, customer_id=customer_id)
    return result

# 定义大模型测试用例并返回
def create_test_case(
    test_data: Dict[str, Any],
    actual_output: str,
    retrieval_context: List[str]
) -> LLMTestCase:
    """
    创建 DeepEval 标准测试用例。

    DeepEval 的大部分指标都以 LLMTestCase 为输入。
    这里把本地测试数据、投顾助手实际回答、检索/背景上下文包装成 DeepEval 可识别的格式。
    """
    return LLMTestCase(
        # 用户原始问题。
        input=test_data["input"],

        # 被测 LLM 应用实际生成的回答。
        actual_output=actual_output,

        # 评估时参考的上下文，尤其会被 HallucinationMetric 用来判断是否存在幻觉。
        retrieval_context=retrieval_context
    )


# ==================== 评估指标定义 ====================

def get_metrics():
    """
    获取 DeepEval 评估指标列表。

    本脚本使用三个指标：
    1. AnswerRelevancyMetric：回答是否切题。
    2. HallucinationMetric：回答是否脱离上下文编造信息。
    3. GEval 自定义指标：投资建议是否考虑客户风险承受能力。
    """

    # 1. 答案相关性指标。
    # threshold=0.6 表示分数达到 0.6 以上才算通过。
    # model="gpt-4o-mini" 表示使用该模型作为评估裁判。
    # include_reason=True 会让 DeepEval 输出评分理由，方便分析失败原因。
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.6,
        model="gpt-4o-mini",
        include_reason=True
    )

    # 2. 幻觉检测指标。
    # 它会比较 actual_output 和 retrieval_context，判断回答是否包含上下文不支持的内容。
    # threshold=0.5 表示幻觉评分需要满足 DeepEval 指标的通过标准。
    hallucination_metric = HallucinationMetric(
        threshold=0.5,
        model="gpt-4o-mini",
        include_reason=True
    )

    # 3. 自定义 GEval 指标：是否考虑客户风险偏好。
    # GEval 适合表达“规则难以精确编码，但可以用自然语言评判”的质量标准。
    risk_consideration_metric = GEval(
        # 指标名称，最终会显示在 DeepEval 的评估结果中。
        name="RiskConsideration",

        # 总体评估标准：要求投资建议考虑客户风险承受能力和投资偏好。
        criteria="评估投资建议是否考虑了客户的风险承受能力和投资偏好",

        # 指定评估模型可以参考哪些测试用例字段。
        # 这里让它看用户输入和实际回答。
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],

        # 更细粒度的评估步骤，指导评估模型如何判断。
        evaluation_steps=[
            "检查回答是否提及风险相关内容",
            "评估建议是否与客户风险等级匹配",
            "判断是否有个性化的考虑"
        ],
        threshold=0.5,
        model="gpt-4o-mini"
    )

    # 返回指标列表，后续 evaluate 会对每个测试用例应用这些指标。
    return [relevancy_metric, hallucination_metric, risk_consideration_metric]


# ==================== 主评估流程 ====================

def run_evaluation():
    """
    运行完整 DeepEval 评估流程。

    流程：
    1. 合并本地测试数据。
    2. 逐条调用投顾助手生成实际回答。
    3. 把实际回答封装成 LLMTestCase。
    4. 加载 DeepEval 指标。
    5. 调用 deepeval.evaluate 执行评估并输出结果。
    """

    print("=" * 60)
    print("DeepEval 投顾AI助手评估")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 合并简单查询和复杂分析查询，形成完整评估集。
    all_test_data = REACTIVE_TEST_CASES + DELIBERATIVE_TEST_CASES

    # DeepEval 需要的 LLMTestCase 列表，后面会逐条构建。
    test_cases = []

    print(f"[准备] 共 {len(all_test_data)} 个测试用例")
    print("-" * 60)

    # 遍历测试数据，运行 Agent 并创建 DeepEval 测试用例。
    for i, test_data in enumerate(all_test_data, 1):
        print(f"\n[{i}/{len(all_test_data)}] 测试: {test_data['input'][:40]}...")

        # 调用投顾助手，得到模型实际回答。
        result = run_agent_and_get_response(
            query=test_data["input"],
            customer_id=test_data["customer_id"]
        )

        # 从投顾助手结果中取最终回答。
        # 如果智能体返回结构变化，这里可能拿不到 final_response。
        actual_output = result.get("final_response", "")

        if not actual_output:
            print(f"  [警告] 未获取到响应，跳过此用例")
            continue

        print(f"  [响应] {actual_output[:100]}...")

        # 读取客户画像，用于把客户风险等级、投资期限等信息加入评估上下文。
        customer_profile = SAMPLE_CUSTOMER_PROFILES.get(test_data["customer_id"], {})

        # retrieval_context 是 DeepEval 幻觉检测和部分质量评估会参考的背景信息。
        # 这里由测试用例自带 context + 客户画像中的关键信息组成。
        retrieval_context = test_data["context"] + [
            f"客户风险等级: {customer_profile.get('risk_tolerance', '未知')}",
            f"投资期限: {customer_profile.get('investment_horizon', '未知')}"
        ]

        # 创建 DeepEval 测试用例。
        test_case = create_test_case(
            test_data=test_data,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)

    print("\n" + "=" * 60)
    print(f"[评估] 开始评估 {len(test_cases)} 个测试用例")
    print("=" * 60)

    # 获取评估指标。
    metrics = get_metrics()
    print(f"[指标] 使用 {len(metrics)} 个评估指标:")
    for m in metrics:
        print(f"  - {m.__class__.__name__}")

    # 调用 DeepEval 执行评估。
    # print_results=True 会在控制台打印每个测试用例的评估分数和原因。
    print("\n[运行] 正在评估...")
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        print_results=True
    )

    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 返回 DeepEval 的评估结果，便于其他脚本复用。
    return results


# ==================== Pytest 风格测试（可选） ====================

def test_reactive_query():
    """
    Pytest 风格的反应式查询测试。

    这个函数可以被 pytest 或 deepeval test run 自动发现并执行。
    它只取 REACTIVE_TEST_CASES 中的第一条用例，适合作为快速单元测试。
    """
    # assert_test 是 DeepEval 在测试框架中使用的断言方法。
    # 如果指标没有通过阈值，assert_test 会让测试失败。
    from deepeval import assert_test

    # 选择第一个简单查询用例。
    test_data = REACTIVE_TEST_CASES[0]

    # 调用投顾助手获取实际输出。
    result = run_agent_and_get_response(
        query=test_data["input"],
        customer_id=test_data["customer_id"]
    )

    # 构造 DeepEval 测试用例。
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=result.get("final_response", ""),
        retrieval_context=test_data["context"]
    )

    # 只使用答案相关性指标做快速判断。
    metric = AnswerRelevancyMetric(threshold=0.6, model="gpt-4o-mini")
    assert_test(test_case, [metric])


def test_deliberative_query():
    """
    Pytest 风格的深思熟虑查询测试。

    这个函数测试复杂投资建议场景，除了回答相关性，也加入幻觉检测。
    与 test_reactive_query 相比，它会额外把客户风险等级加入 retrieval_context。
    """
    from deepeval import assert_test

    # 选择第一个复杂分析类测试用例。
    test_data = DELIBERATIVE_TEST_CASES[0]

    # 调用投顾助手获取实际输出。
    result = run_agent_and_get_response(
        query=test_data["input"],
        customer_id=test_data["customer_id"]
    )

    # 获取客户画像，用于构建更完整的评估上下文。
    customer_profile = SAMPLE_CUSTOMER_PROFILES.get(test_data["customer_id"], {})

    # 构造 DeepEval 测试用例。
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=result.get("final_response", ""),
        retrieval_context=test_data["context"] + [
            f"客户风险等级: {customer_profile.get('risk_tolerance', '未知')}"
        ]
    )

    # 复杂分析类回答同时评估相关性和幻觉风险。
    metrics = [
        AnswerRelevancyMetric(threshold=0.6, model="gpt-4o-mini"),
        HallucinationMetric(threshold=0.5, model="gpt-4o-mini")
    ]
    assert_test(test_case, metrics)


# ==================== 入口 ====================

if __name__ == "__main__":
    # 直接运行本文件时，先检查必要环境变量。
    # DeepEval 的这些指标默认需要 OpenAI 模型作为“评估裁判”，因此必须提供 OPENAI_API_KEY。
    if not os.getenv("OPENAI_API_KEY"):
        print("[错误] 请设置 OPENAI_API_KEY 环境变量")
        print("DeepEval 需要 OpenAI API 作为评估器")
        print()
        print("设置方式 (Windows PowerShell):")
        print('  $env:OPENAI_API_KEY="sk-your-key-here"')
        print()
        print("设置方式 (Linux/Mac):")
        print('  export OPENAI_API_KEY="sk-your-key-here"')
        exit(1)

    # 投顾助手本身使用通义千问/DashScope 作为业务模型，因此还需要 DASHSCOPE_API_KEY。
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("[错误] 请设置 DASHSCOPE_API_KEY 环境变量")
        print("投顾助手需要通义千问 API")
        exit(1)

    # 环境变量检查通过后，运行完整 DeepEval 评估流程。
    run_evaluation()
