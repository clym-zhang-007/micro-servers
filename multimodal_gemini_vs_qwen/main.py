#!/usr/bin/env python
# coding: utf-8

import base64
import mimetypes
import os
import time
from pathlib import Path

from google import genai
from openai import OpenAI
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

# 统一提问，便于不同模型做横向对比
TEXT_PROMPT = "解释一下transformer是什么意思？它是如何工作的？"
IMAGE_PROMPT = "帮我解释下这张照片，描述主体、场景、可能的动作和你不确定的地方。"
VIDEO_PROMPT = "详细描述视频里发生了什么？如果有对话，请把关键对话提取出来。"


def to_data_url(file_path: Path) -> str:
    """将本地文件转成 data URL，给 OpenAI 兼容接口直接传入。"""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    raw = file_path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def safe_text(content) -> str:
    return (content or "").strip()


def run_gemini() -> dict:
    client = genai.Client()
    result = {"text": "", "images": {}, "video": ""}

    # 文本
    text_resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=TEXT_PROMPT,
    )
    result["text"] = safe_text(text_resp.text)
    print("\n===== Gemini 文本结果 =====\n", result["text"])

    # 图片（bike + bike1）
    for image_name in ["bike.png", "bike1.png"]:
        image = Image.open(BASE_DIR / image_name)
        image_resp = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[image, IMAGE_PROMPT],
        )
        result["images"][image_name] = safe_text(image_resp.text)
        print(f"\n===== Gemini 图片结果: {image_name} =====\n", result["images"][image_name])

    # 视频
    print("\n[Gemini] 正在上传视频...")
    video_file = client.files.upload(file=str(BASE_DIR / "video.mp4"))
    print(f"[Gemini] 上传成功: {video_file.name}")

    start_time = time.time()
    while True:
        video_file = client.files.get(name=video_file.name)
        state = video_file.state.name
        elapsed = int(time.time() - start_time)
        print(f"[Gemini][{elapsed:>3}s] 当前状态: {state}")

        if state in ("ACTIVE", "SUCCESS", "SUCCEEDED"):
            print("[Gemini] 视频就绪，开始推理...")
            break
        if state == "FAILED":
            raise ValueError("Gemini 视频处理失败")
        time.sleep(1)

    video_resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[video_file, VIDEO_PROMPT],
    )
    result["video"] = safe_text(video_resp.text)
    print("\n===== Gemini 视频结果 =====\n", result["video"])

    # 如需清理 Gemini 侧上传文件，解除注释：
    # client.files.delete(name=video_file.name)
    return result


def run_qwen_dashscope() -> dict:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("未检测到 DASHSCOPE_API_KEY，请先设置环境变量后再运行。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 你可按需改成 qwen-vl-max / qwen2.5-vl 等 DashScope 已开通的视觉模型
    model_name = "qwen3-vl-flash"
    result = {"text": "", "images": {}, "video": ""}

    # 文本
    text_completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": TEXT_PROMPT}],
    )
    result["text"] = safe_text(text_completion.choices[0].message.content)
    print("\n===== Qwen 文本结果 =====\n", result["text"])

    # 图片（bike + bike1）
    for image_name in ["bike.png", "bike1.png"]:
        image_url = to_data_url(BASE_DIR / image_name)
        image_completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": IMAGE_PROMPT},
                    ],
                }
            ],
        )
        result["images"][image_name] = safe_text(image_completion.choices[0].message.content)
        print(f"\n===== Qwen 图片结果: {image_name} =====\n", result["images"][image_name])

    # 视频
    video_url = to_data_url(BASE_DIR / "video.mp4")
    video_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": VIDEO_PROMPT},
                ],
            }
        ],
    )
    result["video"] = safe_text(video_completion.choices[0].message.content)
    print("\n===== Qwen 视频结果 =====\n", result["video"])

    return result


def compare_brief(gemini_text: str, qwen_text: str) -> str:
    """非常轻量的自动比较：按长度估计信息量，不替代人工判断。"""
    g_len = len(gemini_text)
    q_len = len(qwen_text)
    if g_len > q_len * 1.15:
        winner = "Gemini"
    elif q_len > g_len * 1.15:
        winner = "Qwen"
    else:
        winner = "两者接近"
    return f"- Gemini 字数: {g_len}\n- Qwen 字数: {q_len}\n- 自动判断(仅按信息量粗估): **{winner}**"


def build_markdown(gemini_res: dict, qwen_res: dict) -> str:
    parts = ["# Gemini vs Qwen 多模态结果对比\n"]
    parts.append("## 统一提问\n")
    parts.append(f"- 文本: {TEXT_PROMPT}")
    parts.append(f"- 图片: {IMAGE_PROMPT}")
    parts.append(f"- 视频: {VIDEO_PROMPT}\n")

    parts.append("## 文本问答对比\n")
    parts.append("### Gemini\n")
    parts.append(gemini_res["text"] + "\n")
    parts.append("### Qwen(DashScope)\n")
    parts.append(qwen_res["text"] + "\n")
    parts.append("### 自动比较\n")
    parts.append(compare_brief(gemini_res["text"], qwen_res["text"]) + "\n")

    for image_name in ["bike.png", "bike1.png"]:
        parts.append(f"## 图片理解对比: {image_name}\n")
        parts.append("### Gemini\n")
        parts.append(gemini_res["images"].get(image_name, "") + "\n")
        parts.append("### Qwen(DashScope)\n")
        parts.append(qwen_res["images"].get(image_name, "") + "\n")
        parts.append("### 自动比较\n")
        parts.append(
            compare_brief(
                gemini_res["images"].get(image_name, ""),
                qwen_res["images"].get(image_name, ""),
            )
            + "\n"
        )

    parts.append("## 视频理解对比\n")
    parts.append("### Gemini\n")
    parts.append(gemini_res["video"] + "\n")
    parts.append("### Qwen(DashScope)\n")
    parts.append(qwen_res["video"] + "\n")
    parts.append("### 自动比较\n")
    parts.append(compare_brief(gemini_res["video"], qwen_res["video"]) + "\n")

    parts.append(
        "## 人工结论建议\n"
        "- 自动比较只看字数，不代表准确率。\n"
        "- 建议你重点看：事实准确性、细节完整性、幻觉率、是否忠实于画面/视频内容。\n"
        "- 若要更客观，可再加一轮固定评分模板（准确性/细节/结构/可用性各 10 分）。\n"
    )

    return "\n".join(parts)


def main() -> None:
    gemini_res = run_gemini()
    qwen_res = run_qwen_dashscope()

    report = build_markdown(gemini_res, qwen_res)
    output_md = BASE_DIR / "Gemini-vs-Qwen-output.md"
    output_md.write_text(report, encoding="utf-8")
    print(f"\n已写入 Markdown: {output_md}")


if __name__ == "__main__":
    main()
