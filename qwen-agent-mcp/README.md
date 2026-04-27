# case-mcp

基于 `Qwen-Agent` 与 MCP（Model Context Protocol）的示例集合，演示如何把本地或远程 MCP 工具接入到智能助手中，并通过 `WebUI` 快速启动一个可交互的 Demo。

当前目录包含 3 个示例：

- `assistant_mcp_txt_bot.py`：本地文本文件统计助手，接入本地 `txt_counter.py` MCP 服务
- `assistant_mcp_amap_bot.py`：高德地图助手，接入 `@amap/amap-maps-mcp-server`
- `assistant_mcp_tavily_bot.py`：联网搜索助手，接入 `tavily-mcp`

---

## 目录结构

```text
case-mcp/
├─ assistant_mcp_txt_bot.py
├─ assistant_mcp_amap_bot.py
├─ assistant_mcp_tavily_bot.py
├─ txt_counter.py
├─ requirements.txt
└─ README.md
```

---

## 功能说明

### 1. 文本计数助手

文件：`assistant_mcp_txt_bot.py`

能力：

- 统计桌面 `.txt` 文件数量
- 列出桌面 `.txt` 文件
- 读取指定 `.txt` 文件内容

对应 MCP Server：`txt_counter.py`

### 2. 高德地图助手

文件：`assistant_mcp_amap_bot.py`

能力：

- 地点查询
- 路线规划
- 周边推荐
- 出行辅助

对应 MCP Server：`@amap/amap-maps-mcp-server`

### 3. Tavily 搜索助手

文件：`assistant_mcp_tavily_bot.py`

能力：

- 新闻搜索
- 网络信息检索
- 网页内容提取

对应 MCP Server：`tavily-mcp`

---

## 环境要求

建议使用 Python 3.10+。

部分示例依赖 Node.js / npm / npx：

- `assistant_mcp_amap_bot.py`
- `assistant_mcp_tavily_bot.py`

请确保本机已安装 Node.js，并能正常使用 `npx`。

---

## 依赖安装

在仓库根目录执行：

```bash
pip install -r case-mcp/requirements.txt
```

如果希望更稳定地运行本地 MCP 示例，建议使用独立虚拟环境。

---

## 环境变量配置

### 通用变量

所有示例都需要：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

### 高德地图示例需要

```env
AMAP_MAPS_API_KEY=your_amap_api_key
```

### Tavily 搜索示例需要

```env
TAVILY_API_KEY=your_tavily_api_key
```

---

## 运行方式

### 1. 启动文本计数助手

```bash
python case-mcp/assistant_mcp_txt_bot.py
```

说明：

- 该脚本会自动通过当前 Python 解释器启动本地 `txt_counter.py`
- `txt_counter.py` 提供 MCP 工具能力
- 默认通过 WebUI 启动图形界面

### 2. 启动高德地图助手

```bash
python case-mcp/assistant_mcp_amap_bot.py
```

说明：

- 该脚本通过 `npx` 启动高德地图 MCP Server
- 需要正确配置 `AMAP_MAPS_API_KEY`

### 3. 启动 Tavily 搜索助手

```bash
python case-mcp/assistant_mcp_tavily_bot.py
```

说明：

- 该脚本通过 `npx` 启动 Tavily MCP Server
- 需要正确配置 `TAVILY_API_KEY`

---

## 本地 MCP Server 说明

`txt_counter.py` 是一个基于 `FastMCP` 的本地 MCP Server，提供以下工具：

- `count_desktop_txt_files`
- `list_desktop_txt_files`
- `read_txt_file`

如果想单独验证它，也可以直接运行：

```bash
python case-mcp/txt_counter.py
```

---

## 常见问题

### 1. 启动时报 `Connection closed`

通常是以下原因之一：

- 调用了错误的 Python 环境
- 本地 MCP Server 脚本路径不正确
- 当前环境缺少 `mcp` / `fastmcp` / `qwen_agent`

当前 `assistant_mcp_txt_bot.py` 已改为使用当前解释器绝对路径来启动 `txt_counter.py`，可减少这类问题。

### 2. `npx` 命令不可用

请先安装 Node.js，并确认命令行中可执行：

```bash
npx -v
```

### 3. 地图或搜索助手无法调用工具

请检查：

- 对应 API Key 是否已配置
- 网络是否可访问外部服务
- `npx` 是否可正常安装并运行对应 MCP Server

### 4. WebUI 启动失败

请优先检查：

- `DASHSCOPE_API_KEY` 是否正确
- Python 依赖是否安装完整
- MCP Server 是否成功启动

---

## 适合用于什么场景

这个目录里的示例非常适合作为以下用途：

- 学习如何将 MCP 工具接入 Agent
- 快速搭建一个带工具能力的助手原型
- 作为团队内部 `Agent + MCP` 模板的起点
- 用于沉淀后续 Skill / Harness 体系中的 MCP pattern

---

## 版本控制建议

建议排除以下内容：

- 真实 API Key
- 本地 `.env` 文件
- 个人测试产生的缓存或临时文件

如需团队协作，可进一步补充：

- `.env.example`
- 更统一的配置文件结构
- 按功能拆分的模块化代码
