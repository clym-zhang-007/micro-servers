# multimodal_gemini_vs_qwen

对比 `Gemini -- gemini-3-flash-preview` 与 `Qwen(DashScope) -- qwen3-vl-flash` 在多模态任务（文本、图片、视频）上的输出效果。

## 功能说明

- 使用同一组提示词，分别调用两种模型。
- 同时处理以下输入：
  - 文本问题
  - 图片 `bike.png`、`bike1.png`
  - 视频 `video.mp4`
- 自动生成对比报告：`Gemini-vs-Qwen-output.md`。

## 目录结构

```text
multimodal_gemini_vs_qwen/
├─ main.py
├─ requirements.txt
├─ bike.png
├─ bike1.png
├─ video.mp4
├─ Gemini-vs-Qwen-output.md   # 运行后生成/更新
└─ README.md
```

## 环境准备

```bash
python -m pip install --upgrade pip
python -m pip install -r multimodal_gemini_vs_qwen/requirements.txt
```

## 环境变量

运行前请设置：

- `GEMINI_API_KEY`：用于 Gemini
- `DASHSCOPE_API_KEY`：用于 DashScope Qwen

PowerShell 示例：

```powershell
$env:GOOGLE_API_KEY="your_gemini_api_key"
$env:DASHSCOPE_API_KEY="your_dashscope_api_key"
```

## 运行方式

在仓库根目录执行：

```bash
python multimodal_gemini_vs_qwen/main.py
```

执行完成后，查看：

- `multimodal_gemini_vs_qwen/Gemini-vs-Qwen-output.md`

## 当前模型配置

`main.py` 默认配置：

- Gemini: `gemini-3-flash-preview`
- Qwen: `qwen3-vl-flash`

你可以在 `main.py` 中按账号开通情况修改模型名称。

## 对比结论建议

脚本内置了一个“按文本长度粗略比较”的自动判断，仅用于快速参考。  
真正评估建议关注：

- 事实准确性
- 细节完整性
- 幻觉率
- 是否忠实于图像/视频内容

