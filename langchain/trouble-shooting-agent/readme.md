# 网络故障诊断 Agent

基于 LangChain Agent + 通义千问（Qwen）的网络故障诊断示例，演示如何自定义多个诊断工具并交由 Agent 自主调度。

## 内置工具

| 工具 | 功能 |
|---|---|
| `ping_tool` | 检查到指定主机/ IP 的网络连通性 |
| `dns_tool` | 模拟 DNS 解析，获取主机名对应的 IP 地址 |
| `interface_check_tool` | 检查本机网络接口状态（IP、是否启用） |
| `log_analysis_tool` | 搜索系统日志中与网络问题相关的错误条目 |

> 注：当前工具为**模拟实现**，实际生产环境可替换为真实系统命令或 API 调用。

## 安装依赖

```bash
pip install -r requirements.txt
```

需要设置环境变量 `DASHSCOPE_API_KEY`（通义千问 API Key）。

## 运行

```bash
python 2-network_diagnosis_agent.py
```

运行后会依次执行两个预设的诊断任务示例，演示 Agent 如何自主调用工具进行故障排查。
