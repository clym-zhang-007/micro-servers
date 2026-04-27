# LangChain 工具链示例

基于 LangChain 生态的多工具任务链实践，展示两种不同的工具编排方式。

## 示例说明

| 文件 | 方式 | 特点 |
|---|---|---|
| `1-simple_toolchain.py` | Agent + Tool Calling | 使用 `create_agent` + `@tool`，由 LLM 自主决策调用哪个工具 |
| `2-simple_toolchain.py` | LCEL 管道编排 | 使用 `RunnableLambda` / `RunnableMap` / `RunnablePassthrough`，声明式组合工具链 |

### 内置工具

- **文本分析**：统计字数、字符数，判断情感倾向（积极/消极/中性）
- **数据转换**：JSON ↔ CSV 格式互转
- **文本处理**：查找、替换、统计行数

## 安装依赖

```bash
pip install -r requirements.txt
```

需要设置环境变量 `DASHSCOPE_API_KEY`（通义千问 API Key）。

## 运行

```bash
python 1-simple_toolchain.py   # Agent 模式
python 2-simple_toolchain.py   # LCEL 模式
```
