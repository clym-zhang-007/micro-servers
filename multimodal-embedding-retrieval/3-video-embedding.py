#!/usr/bin/env python
# coding: utf-8

# 多模态向量化模型目前仅支持以URL形式输入视频文件，暂不支持直接传入本地视频。
import dashscope
import json
from http import HTTPStatus
from pathlib import Path

base_dir = Path(__file__).resolve().parent
# 实际使用中请将url地址替换为您的视频url地址
video_path = base_dir / "disney_knowledge_base" / "videos" / "video.mp4"
# dashscope 这里需要字符串；Windows 下 as_uri() 可能导致 /E:/... 路径解析异常
video = str(video_path.resolve())
input = [{'video': video}]
# 调用模型接口
resp = dashscope.MultiModalEmbedding.call(
    model="tongyi-embedding-vision-plus",
    input=input
)

if resp.status_code == HTTPStatus.OK:
    result = {
        "status_code": resp.status_code,
        "request_id": getattr(resp, "request_id", ""),
        "code": getattr(resp, "code", ""),
        "message": getattr(resp, "message", ""),
        "output": resp.output,
        "usage": resp.usage
    }
    print(json.dumps(result, ensure_ascii=False, indent=4))

