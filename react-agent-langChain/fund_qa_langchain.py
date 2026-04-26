#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
私募基金运作指引问答助手 - ReAct 智能体实现

使用 langchain classic agent 保留思考-执行-观察循环，
并避免当前 langchain 新版 agent 工厂链路的兼容问题。
"""

import os
import re
from typing import Any, List, Optional

import jieba
from langchain_community.chat_models import ChatTongyi
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, tool

# 通义千问API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 简化的私募基金规则数据库
FUND_RULES_DB = [
    {
        "id": "rule001",
        "category": "设立与募集",
        "question": "私募基金的合格投资者标准是什么？",
        "answer": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于100万元且符合下列条件之一的单位和个人：\n1. 净资产不低于1000万元的单位\n2. 金融资产不低于300万元或者最近三年个人年均收入不低于50万元的个人"
    },
    {
        "id": "rule002",
        "category": "设立与募集",
        "question": "私募基金的最低募集规模要求是多少？",
        "answer": "私募证券投资基金的最低募集规模不得低于人民币1000万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。"
    },
    {
        "id": "rule014",
        "category": "监管规定",
        "question": "私募基金管理人的风险准备金要求是什么？",
        "answer": "私募证券基金管理人应当按照管理费收入的10%计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。"
    }
]


class PromptDebugHandler(BaseCallbackHandler):
    """打印每一轮真正发送给 LLM 的 prompt。"""

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        for idx, prompt in enumerate(prompts, start=1):
            print(f"\n===== 第 {idx} 次发送给 LLM 的 Prompt =====")
            print(prompt)
            print("===== Prompt 结束 =====\n")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        generations = response.generations or []
        if generations and generations[0]:
            text = getattr(generations[0][0], "text", None)
            if not text and hasattr(generations[0][0], "message"):
                text = getattr(generations[0][0].message, "content", None)
            print("===== LLM 本轮输出 =====")
            print(text or "<empty>")
            print("===== 输出结束 =====\n")


def tokenize_text(text: str) -> set[str]:
    """使用 jieba 对中文文本分词，并过滤空白词。"""
    return {word.strip().lower() for word in jieba.cut(text) if word.strip()}


@tool
def search_rules_by_keywords(keywords: str) -> str:
    """通过关键词搜索相关私募基金规则。输入应为相关关键词，多个关键词用逗号或空格分隔。"""

    keywords = keywords.strip().lower()
    keyword_list = [kw for kw in re.split(r'[,，\s]+', keywords) if kw]

    matched_rules = []
    for rule in FUND_RULES_DB:
        rule_text = (rule["category"] + " " + rule["question"]).lower()
        match_count = sum(1 for kw in keyword_list if kw in rule_text)
        if match_count > 0:
            matched_rules.append((rule, match_count))

    matched_rules.sort(key=lambda x: x[1], reverse=True)

    if not matched_rules:
        return "未找到与关键词相关的规则。"

    result = []
    for rule, _ in matched_rules[:2]:
        result.append(f"类别: {rule['category']}\n问题: {rule['question']}\n答案: {rule['answer']}")

    return "\n\n".join(result)


@tool
def search_rules_by_category(category: str) -> str:
    """根据规则类别查询私募基金规则。输入应为类别名称，可选类别：设立与募集、监管规定。"""
    category = category.strip()
    matched_rules = []

    for rule in FUND_RULES_DB:
        if category.lower() in rule["category"].lower():
            matched_rules.append(rule)

    if not matched_rules:
        return f"未找到类别为 '{category}' 的规则。"

    result = []
    for rule in matched_rules:
        result.append(f"问题: {rule['question']}\n答案: {rule['answer']}")

    return "\n\n".join(result)


@tool
def answer_question(query: str) -> str:
    """在知识库中搜索并回答用户关于私募基金的问题。输入应为完整的用户问题。"""
    query = query.strip()

    best_rule = None
    best_score = 0
    query_words = tokenize_text(query)

    for rule in FUND_RULES_DB:
        # 拆成词集合
        rule_words = tokenize_text(rule["question"] + " " + rule["category"])
        # 两者的交集--共同包含的词
        common_words = query_words.intersection(rule_words)

        # 这个分数越大，说明：用户问题里的词，有越多能在当前这条规则里找到匹配。
        score = len(common_words) / max(1, len(query_words))
        if score > best_score:
            best_score = score
            best_rule = rule

    if best_score < 0.2 or best_rule is None:
        return "在知识库中未找到与该问题直接相关的信息。请尝试使用关键词搜索或类别查询。"

    return f"根据知识库信息：\n\n类别: {best_rule['category']}\n问题: {best_rule['question']}\n答案: {best_rule['answer']}"


def build_prompt() -> PromptTemplate:
    # {input} 的来源是在 invoke() 里传进去的键值
    return PromptTemplate.from_template(
        """你是一个私募基金问答助手，专门回答关于私募基金规则和运作的问题。

你可以使用以下工具来查询信息：
{tools}

请严格按照以下格式思考和作答：
Question: 用户的问题
Thought: 你要先思考应该使用哪个工具
Action: 要调用的工具名称，必须是 [{tool_names}] 之一
Action Input: 传给工具的输入
Observation: 工具返回的结果
...（Thought/Action/Action Input/Observation 可以重复多轮）
Thought: 我已经得到足够信息，可以回答用户了
Final Answer: 给用户的最终回答

要求：
1. 优先使用工具获取信息，不要凭空编造。
2. 如果知识库中没有相关信息，请明确说明未找到。
3. 回答要专业、简洁、准确。
4. 最终回答使用中文。

Question: {input}
Thought:{agent_scratchpad}
"""
    )


def print_prompt_debug(prompt: PromptTemplate, tools: list, user_input: str, agent_scratchpad: str = "") -> None:
    rendered_prompt = prompt.format(
        tools=render_text_description(tools),
        tool_names=", ".join(tool.name for tool in tools),
        input=user_input,
        agent_scratchpad=agent_scratchpad,
    )
    print("\n===== 发送给 LLM 的 Prompt（本轮起始） =====")
    print(rendered_prompt)
    print("===== Prompt 结束 =====\n")


def create_fund_qa_agent() -> tuple[AgentExecutor, PromptTemplate, list]:
    llm = ChatTongyi(
        model="qwen-plus",
        dashscope_api_key=DASHSCOPE_API_KEY,
        callbacks=[PromptDebugHandler()],
    )
    tools = [search_rules_by_keywords, search_rules_by_category, answer_question]
    prompt = build_prompt()

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor, prompt, tools


if __name__ == "__main__":
    fund_qa_agent, agent_prompt, agent_tools = create_fund_qa_agent()

    print("=== 私募基金运作指引问答助手（ReAct 智能体）===\n")
    print("使用模型：qwen-plus")
    print("您可以提问关于私募基金的各类问题，输入'退出'结束对话\n")

    while True:
        user_input = input("请输入您的问题：")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("感谢使用，再见！")
            break

        print_prompt_debug(agent_prompt, agent_tools, user_input)
        response = fund_qa_agent.invoke({"input": user_input})
        print(f"回答: {response['output']}\n")
        print("-" * 40)
