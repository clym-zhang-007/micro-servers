#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangSmith 测试与评估模块。

功能概览：
1. 动态加载同目录下的投顾智能体主程序。
2. 创建或复用 LangSmith Dataset。
3. 定义自动评估器，对智能体输出打分。
4. 执行批量评估，并把结果上传到 LangSmith。
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib.util
import sys

from langsmith import Client, RunEvaluator  # Client 访问 LangSmith；RunEvaluator 是自定义评估器基类
from langsmith.evaluation import evaluate  # evaluate 负责遍历数据集并执行批量评估
from langsmith.schemas import Example, Run  # Example 是测试样例；Run 是一次被测函数运行记录

# 当前脚本目录，用来定位同目录下的被测智能体文件。
CURRENT_DIR = Path(__file__).resolve().parent  # 当前脚本目录
# 被测智能体脚本路径；文件名以数字开头，所以不能直接用普通 import 导入。
HYBRID_AGENT_PATH = CURRENT_DIR / "1-hybrid_wealth_advisor_langgraph_langsmith.py"

# 按文件路径创建模块加载规范，后续用它动态执行被测智能体代码。
spec = importlib.util.spec_from_file_location(
    "hybrid_wealth_advisor_langgraph_langsmith",
    HYBRID_AGENT_PATH,
)
# 如果加载器创建失败，说明目标文件路径或模块规范有问题，需要提前中断。
if spec is None or spec.loader is None:
    raise ImportError(f"无法为主智能体脚本创建模块加载器: {HYBRID_AGENT_PATH}")

# 创建模块对象；此时只是创建对象，还没有执行目标文件中的代码。
hybrid_module = importlib.util.module_from_spec(spec)
# 注册到 sys.modules，让后续导入语句可以找到这个动态模块。
sys.modules["hybrid_wealth_advisor_langgraph_langsmith"] = hybrid_module
# 执行目标文件中的代码，完成模块初始化。
spec.loader.exec_module(hybrid_module)

# 导入本评估脚本要调用的投顾智能体入口函数 --- 直接调用被测智能体代码。
from hybrid_wealth_advisor_langgraph_langsmith import run_wealth_advisor

# 读取 LangSmith API Key，用于创建数据集、写入样例和上传评估结果。
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
# 判断是否启用 LangSmith tracing；LangChain/LangSmith 通常使用该变量控制追踪开关。
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

# 如果没有启用 tracing 或缺少 API Key，就不初始化 LangSmith 客户端。
# 后续函数会根据 client 是否为 None 来决定是否继续执行。
if not LANGSMITH_ENABLED or not LANGSMITH_API_KEY:
    print("警告: LangSmith 未启用，无法进行测试和评估")
    print("请设置环境变量:")
    print("  - LANGSMITH_API_KEY: 您的 API 密钥")
    print("  - LANGCHAIN_TRACING_V2: true")
    client = None
else:
    # 初始化 LangSmith 客户端 --- 创建 LangSmith 客户端对象。
    client = Client(api_key=LANGSMITH_API_KEY)
    print("[LangSmith] 客户端已初始化，可以开始测试和评估")


# 结构化测试用例：按处理模式分组（reactive / deliberative / edge），每组包含输入和预期输出
REACTIVE_TEST_CASES = [
    {"inputs": {"user_query": "今天上证指数的表现如何？", "customer_id": "customer1"}, "expected_outputs": {"processing_mode": "reactive", "should_contain": ["上证指数", "点位", "涨跌"]}},
    {"inputs": {"user_query": "我的投资组合中科技股占比是多少？", "customer_id": "customer1"}, "expected_outputs": {"processing_mode": "reactive", "should_contain": ["科技", "占比", "投资组合"]}},
    {"inputs": {"user_query": "请解释一下什么是ETF？", "customer_id": "customer1"}, "expected_outputs": {"processing_mode": "reactive", "should_contain": ["ETF", "基金", "交易"]}},
]

# deliberative 测试用例：测试复杂规划、投资策略评估等需要深度分析的问题。
# 这些问题期望智能体进入 deliberative 模式，并在回答中覆盖关键主题。
DELIBERATIVE_TEST_CASES = [
    {"inputs": {"user_query": "根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？", "customer_id": "customer1"}, "expected_outputs": {"processing_mode": "deliberative", "should_contain": ["投资组合", "调整", "经济衰退", "建议"]}},
    {"inputs": {"user_query": "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。", "customer_id": "customer1"}, "expected_outputs": {"processing_mode": "deliberative", "should_contain": ["退休", "投资策略", "评估", "建议"]}},
    {"inputs": {"user_query": "我想为子女准备教育金，请帮我设计一个10年期的投资计划。", "customer_id": "customer1"}, "expected_outputs": {"processing_mode": "deliberative", "should_contain": ["教育金", "10年", "投资计划", "建议"]}},
]

# 边界测试用例：测试空输入和超长输入是否会让程序崩溃。
# should_handle_error / should_handle 是自定义标记，不是 LangSmith 内置参数。
EDGE_CASE_TEST_CASES = [
    {"inputs": {"user_query": "", "customer_id": "customer1"}, "expected_outputs": {"should_handle_error": True}},
    {"inputs": {"user_query": "这是一个非常长的查询" * 100, "customer_id": "customer1"}, "expected_outputs": {"should_handle": True}},
]

# 汇总全部测试用例，后续统一写入 LangSmith Dataset。
ALL_TEST_CASES = REACTIVE_TEST_CASES + DELIBERATIVE_TEST_CASES + EDGE_CASE_TEST_CASES


def _get_example_inputs(example):
    """从 LangSmith Example 或普通 dict 中提取 inputs，兼容不同版本的传参格式。"""
    try:
        if hasattr(example, "inputs") and example.inputs is not None:
            return example.inputs if isinstance(example.inputs, dict) else {}
        if isinstance(example, dict):
            if "inputs" in example:
                return example["inputs"] if isinstance(example["inputs"], dict) else {}
            if "user_query" in example:
                return example
    except Exception:
        pass
    return {}


def _get_example_outputs(example):
    """从 LangSmith Example 或普通 dict 中提取 expected outputs，供评估器读取。"""
    try:
        if hasattr(example, "outputs") and example.outputs is not None:
            return example.outputs if isinstance(example.outputs, dict) else {}
        if isinstance(example, dict) and "outputs" in example:
            return example["outputs"] if isinstance(example["outputs"], dict) else {}
    except Exception:
        pass
    return {}


class ProcessingModeEvaluator(RunEvaluator):
    """评估智能体选择的 processing_mode 是否符合测试样例预期。"""
    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        try:
            # 从测试样例的 expected_outputs 中读取期望的处理模式。
            expected_mode = _get_example_outputs(example).get("processing_mode")
            # 如果没有配置期望模式，说明该样例不参与本评估项打分。
            if not expected_mode:
                return {"key": "processing_mode", "score": None, "comment": "未指定期望的处理模式"}
            # 从被测函数实际输出中读取处理模式。
            actual_mode = run.outputs.get("processing_mode") if run.outputs else None
            # 实际模式与期望模式完全一致则给满分。
            if actual_mode == expected_mode:
                return {"key": "processing_mode", "score": 1.0, "comment": f"处理模式正确: {actual_mode}"}
            # 不匹配则给 0 分，并在 comment 中记录差异。
            return {"key": "processing_mode", "score": 0.0, "comment": f"处理模式不匹配: 期望 {expected_mode}, 实际 {actual_mode}"}
        except Exception as e:
            return {"key": "processing_mode", "score": 0, "comment": f"评估错误: {str(e)}"}


class ResponseCompletenessEvaluator(RunEvaluator):
    """通过关键词命中率评估 final_response 是否覆盖了预期信息点。"""
    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        try:
            # 如果没有最终回答字段，则认为该样例没有产生有效响应。
            if not run.outputs or "final_response" not in run.outputs:
                return {"key": "response_completeness", "score": 0, "comment": "无响应输出"}
            # 读取被测函数返回的最终回答文本。
            response = run.outputs.get("final_response", "")
            # should_contain 是测试样例里自定义的关键词列表，不是 LangSmith 内置字段。
            expected_keywords = _get_example_outputs(example).get("should_contain", [])
            # 如果没有配置关键词，说明该样例不参与回答完整性评分。
            if not expected_keywords:
                return {"key": "response_completeness", "score": None, "comment": "未指定期望的关键词"}
            # 统一转小写，便于英文关键词做大小写无关匹配；中文不受影响。
            response_lower = response.lower()
            # 统计实际回答中命中的关键词。
            found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
            # 完整性得分 = 命中关键词数 / 期望关键词总数。
            completeness = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
            return {"key": "response_completeness", "score": completeness, "comment": f"找到 {len(found_keywords)}/{len(expected_keywords)} 个期望关键词"}
        except Exception as e:
            return {"key": "response_completeness", "score": 0, "comment": f"评估错误: {str(e)}"}


def create_test_dataset(dataset_name: str = "wealth-advisor-test-dataset") -> Optional[str]:
    """创建或复用 LangSmith Dataset 测试数据集 ， 并把本地测试用例写入为 Example。"""
    if not client:
        print("错误: LangSmith 客户端未初始化")
        return None
    try:
        try:
            # 先尝试从langsmith读取数据集；如果能读到，说明数据集已经存在。
            client.read_dataset(dataset_name=dataset_name)
            print(f"[LangSmith] 数据集已存在: {dataset_name}")
            # 拉取已有样例数量，用于判断是否需要重复写入测试用例。 --- 已存在的样例list
            existing_examples = list(client.list_examples(dataset_name=dataset_name))
            print(f"  现有测试用例数量: {len(existing_examples)}")
            # 如果数据集已有样例，直接复用，避免重复创建相同测试数据。
            if len(existing_examples) > 0:
                # 如果存在就直接返回，避免重复创建相同测试数据。
                return dataset_name
        except Exception:
            # 读取失败通常表示数据集不存在，因此创建新的 LangSmith Dataset。
            print("[LangSmith] 数据集不存在，正在创建...")
            client.create_dataset(dataset_name=dataset_name, description="投顾AI助手测试数据集")
            print(f"[LangSmith] 数据集已创建: {dataset_name}")

        # 遍历本地测试用例，把每条用例写成 LangSmith Example。
        for test_case in ALL_TEST_CASES:
            try:
                # inputs 是被测函数输入；outputs 是评估器读取的期望输出。
                client.create_example(inputs=test_case["inputs"], outputs=test_case.get("expected_outputs", {}), dataset_name=dataset_name)
            except Exception as e:
                error_msg = str(e)
                # 重复样例不影响整体评估流程，直接跳过。
                if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                    continue
                print(f"警告: 添加测试用例失败: {error_msg}")

        return dataset_name
    except Exception as e:
        print(f"创建测试数据集失败: {str(e)}")
        return None


def run_evaluation(dataset_name: str, experiment_name: Optional[str] = None, evaluators: Optional[List[RunEvaluator]] = None):
    """运行 LangSmith 批量评估，并返回 evaluate 的执行结果。"""
    if not client:
        print("错误: LangSmith 客户端未初始化")
        return None
    # 如果没有传实验名称，就用时间戳生成一个唯一实验名。
    if not experiment_name:
        experiment_name = f"wealth-advisor-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # 默认使用两个规则评估器：处理模式评估 + 回答关键词完整性评估。
    if not evaluators:
        evaluators = [ProcessingModeEvaluator(), ResponseCompletenessEvaluator()]

    # 定义测试函数，用于执行投顾智能体，并返回评估结果。 --- 被测函数
    def test_function(example: Example) -> Dict[str, Any]:
        try:
            # 提取测试样例中的输入字段。
            example_inputs = _get_example_inputs(example)
            # user_query 是用户问题，customer_id 用于模拟不同客户画像。
            user_query = example_inputs.get("user_query", "")
            customer_id = example_inputs.get("customer_id", "customer1")
            # 空问题直接返回结构化错误，避免进入智能体主流程后出现不可控异常。
            if not user_query or (isinstance(user_query, str) and not user_query.strip()):
                return {"error": "用户查询为空", "final_response": "用户查询为空，无法处理", "processing_mode": "unknown", "query_type": "unknown"}
            # 调用真实的投顾智能体入口函数，获得业务输出。
            result = run_wealth_advisor(user_query=user_query, customer_id=customer_id)
            return {
                "final_response": result.get("final_response", ""),
                "processing_mode": result.get("processing_mode", "unknown"),
                "query_type": result.get("query_type", "unknown"),
                "error": result.get("error"),
            }
        except Exception as e:
            return {"error": str(e), "final_response": f"执行错误: {str(e)}", "processing_mode": "unknown", "query_type": "unknown"}

    # 执行 LangSmith 批量评估。max_concurrency=1 表示串行执行，便于调试并避免触发模型限流。
    return evaluate(test_function, data=dataset_name, evaluators=evaluators, experiment_prefix=experiment_name, max_concurrency=1)


if __name__ == "__main__":
    # 直接运行本文件时，从这里开始执行完整评估流程。
    print("=" * 60)
    print("LangSmith 测试与评估工具")
    print("=" * 60)
    print()

    if not LANGSMITH_ENABLED or not client:
        print("❌ 错误: LangSmith 未启用，无法进行测试和评估")
        exit(1)

    dataset_name = "wealth-advisor-test-dataset"
    created_dataset = create_test_dataset(dataset_name)
    if not created_dataset:
        print("❌ 数据集准备失败")
        exit(1)

    experiment_name = f"wealth-advisor-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"实验名称: {experiment_name}")
    print(f"数据集: {dataset_name}")
    print("开始运行评估，这可能需要几分钟时间，请耐心等待...")

    results = run_evaluation(dataset_name, experiment_name)
    if results:
        print("✓ 评估完成！")
        print(f"实验名称: {experiment_name}")
        print("查看详细结果: https://smith.langchain.com")
    else:
        print("⚠ 评估完成，但未返回结果")
