#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangSmith 测试与评估 - 快速开始示例。

这个脚本是一个“入口示例脚本”，用于演示如何调用 2-langsmith_testing_evaluation.py
中封装好的 LangSmith 测试与评估能力。

它主要完成三件事：
1. 检查 LangSmith 所需环境变量是否已经配置。
2. 创建或复用 LangSmith 测试数据集。
3. 演示单条测试，并可选择运行完整批量评估。
"""

import os

# importlib.util 用于按文件路径动态导入 Python 文件。
# 因为 2-langsmith_testing_evaluation.py 文件名以数字开头，不能直接写：
# import 2-langsmith_testing_evaluation
# 所以这里需要使用 importlib 手动加载。
import importlib.util

# sys.modules 是 Python 已加载模块的注册表。
# 把动态加载出来的模块放进去后，后面就可以用普通 import 语句导入其中的函数。
import sys

# 创建模块加载规范。
# 第一个参数 "langsmith_testing_evaluation" 是给这个动态模块起的临时模块名。
# 第二个参数 "2-langsmith_testing_evaluation.py" 是要加载的实际文件路径。
spec = importlib.util.spec_from_file_location(
    "langsmith_testing_evaluation",
    "2-langsmith_testing_evaluation.py"
)

# 根据加载规范创建模块对象。
# 注意：这一步只是创建模块对象，还没有真正执行目标文件中的代码。
eval_module = importlib.util.module_from_spec(spec)

# 把动态模块注册到 sys.modules。
# 这样后面的 from langsmith_testing_evaluation import ... 才能正常工作。
sys.modules["langsmith_testing_evaluation"] = eval_module

# 执行 2-langsmith_testing_evaluation.py 文件中的代码，完成模块初始化。
# 执行完成后，该文件中定义的函数、类、变量就会挂到 eval_module 上。
spec.loader.exec_module(eval_module)

# 从动态加载的评估模块中导入需要使用的函数。
# create_test_dataset：创建或复用 LangSmith Dataset。
# run_evaluation：运行完整批量评估。
# run_wealth_advisor：真实的投顾智能体入口函数，用于本脚本中的单条测试。
from langsmith_testing_evaluation import (
    create_test_dataset,
    run_evaluation,
    run_wealth_advisor
)


def run_single_test(user_query: str, customer_id: str = "customer1"):
    """
    运行单条投顾智能体测试。

    这个函数是快速开始脚本中的本地辅助函数，用于在执行完整 LangSmith 评估前，
    先用一条输入验证智能体主链路是否能正常运行。
    """
    # 调用真实的投顾智能体入口函数。
    result = run_wealth_advisor(user_query=user_query, customer_id=customer_id)

    # 打印关键输出，方便用户在命令行中快速确认结果。
    print(f"处理模式: {result.get('processing_mode', 'unknown')}")
    print(f"查询类型: {result.get('query_type', 'unknown')}")
    print(f"最终回答: {result.get('final_response', '')}")

    # 返回完整结果，便于后续扩展断言或调试。
    return result


def main():
    """
    主函数：演示完整的 LangSmith 快速测试流程。

    执行顺序：
    1. 检查 LangSmith 环境变量。
    2. 创建测试数据集。
    3. 运行一个单条测试样例。
    4. 根据用户输入决定是否运行完整评估。
    """

    # 打印标题分隔线，让命令行输出更清晰。
    print("=" * 60)
    print("LangSmith 测试与评估 - 快速开始示例")
    print("=" * 60)
    print()

    # 检查 LangSmith 是否启用。
    # LANGSMITH_API_KEY：访问 LangSmith 服务所需的 API Key。
    # LANGCHAIN_TRACING_V2=true：开启 LangChain/LangSmith tracing。
    # 两者缺一不可，否则后续无法把数据集和评估结果上传到 LangSmith。
    if not os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_TRACING_V2", "").lower() != "true":
        print("❌ 错误: LangSmith 未启用")
        print()
        print("请先设置环境变量:")
        print("  $env:LANGSMITH_API_KEY='your-api-key'")
        print("  $env:LANGCHAIN_TRACING_V2='true'")
        print()
        return

    # 走到这里说明环境变量检查通过，可以继续执行 LangSmith 相关操作。
    print("✓ LangSmith 已启用")
    print()

    # 步骤 1：创建或复用测试数据集。
    # 数据集会保存在 LangSmith 远端，用于后续 evaluate 批量读取测试样例。
    print("步骤 1: 创建测试数据集...")
    print("-" * 60)

    # Dataset 名称需要和 2-langsmith_testing_evaluation.py 中使用的名称保持一致。
    # 如果 LangSmith 中已经存在同名数据集，则 create_test_dataset 会尝试复用。
    dataset_name = "wealth-advisor-test-dataset"

    try:
        # 调用评估模块中的 create_test_dataset。
        # 该函数内部会检查数据集是否存在，不存在则创建，并写入本地定义的测试用例。
        created_dataset = create_test_dataset(dataset_name)
        if created_dataset:
            print(f"✓ 数据集创建成功: {dataset_name}")
        else:
            # 返回空值通常表示客户端未初始化、创建失败，或者已有数据集复用逻辑没有返回名称。
            print("⚠ 数据集可能已存在，继续使用现有数据集")
    except Exception as e:
        # 数据集创建失败不一定代表后续无法继续。
        # 如果远端已经存在同名数据集，仍然可以尝试继续评估。
        print(f"⚠ 数据集创建警告: {str(e)}")
        print("  继续使用现有数据集（如果存在）")

    print()

    # 步骤 2：运行单个测试用例。
    # 这一步是轻量级 smoke test，用于快速确认智能体调用链是否正常。
    print("步骤 2: 运行单个测试用例（演示）...")
    print("-" * 60)

    # 这里选择一个典型的 reactive 查询作为演示。
    test_query = "今天上证指数的表现如何？"
    print(f"测试查询: {test_query}")

    try:
        # run_single_test 预期会调用投顾智能体，并输出或返回单次测试结果。
        # 第二个参数 "customer1" 表示使用 customer1 这个模拟客户画像。
        result = run_single_test(test_query, "customer1")
        print("✓ 单个测试完成")
    except Exception as e:
        # 单条测试失败时不直接退出，仍然允许用户选择是否继续完整评估。
        print(f"⚠ 单个测试失败: {str(e)}")

    print()

    # 步骤 3：询问是否运行完整评估。
    # 完整评估会遍历 LangSmith Dataset 中的所有测试样例，并调用 evaluator 自动打分。
    print("步骤 3: 运行完整评估")
    print("-" * 60)
    print("注意: 完整评估会运行所有测试用例，可能需要较长时间")

    # 通过命令行交互决定是否继续，避免用户误触发大量 LLM/API 调用。
    user_input = input("是否运行完整评估？(y/n): ").strip().lower()

    if user_input == 'y':
        # 为本次评估生成唯一实验名称。
        # 时间戳可以帮助你在 LangSmith Experiments 页面区分不同评估批次。
        experiment_name = f"wealth-advisor-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"\n开始运行评估...")
        print(f"实验名称: {experiment_name}")
        print("这可能需要几分钟时间，请耐心等待...")
        print()

        try:
            # 调用评估模块中的 run_evaluation。
            # 它会内部调用 langsmith.evaluation.evaluate：
            # 1. 读取 dataset_name 对应的数据集。
            # 2. 针对每条 Example 调用被测函数。
            # 3. 使用 evaluator 计算分数。
            # 4. 把实验结果上传到 LangSmith。
            results = run_evaluation(dataset_name, experiment_name)
            if results:
                print()
                print("=" * 60)
                print("✓ 评估完成！")
                print("=" * 60)
                print()
                print("查看结果:")
                print(f"  https://smith.langchain.com")
                print()
                print("在 LangSmith 界面中:")
                print("  1. 进入 'Experiments' 页面")
                print(f"  2. 查找实验: {experiment_name}")
                print("  3. 查看详细的评估结果和分数")
            else:
                # evaluate 没返回结果不一定代表远端没有记录，可能只是 SDK 返回值为空。
                print("⚠ 评估完成，但未返回结果")
        except Exception as e:
            # 完整评估通常涉及远端 API、LLM 调用、数据集读取等多个环节，因此统一捕获异常。
            print(f"❌ 评估失败: {str(e)}")
    else:
        # 用户选择 n 或其他输入时跳过完整评估。
        print("跳过完整评估")

    # 打印结束信息和后续使用提示。
    print()
    print("=" * 60)
    print("完成！")
    print("=" * 60)
    print()
    print("提示:")
    print("  - 查看详细使用说明: 阅读 LANGSMITH_TESTING_GUIDE.md")
    print("  - 运行完整工具: python langsmith_testing_evaluation.py")
    print("  - 访问 LangSmith: https://smith.langchain.com")


if __name__ == "__main__":
    # datetime 只在用户选择运行完整评估时用于生成实验名称。
    # 放在入口处导入可以避免脚本作为模块被导入时产生不必要依赖。
    from datetime import datetime

    # 执行主流程。
    main()
