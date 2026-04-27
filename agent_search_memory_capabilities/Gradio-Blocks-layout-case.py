# -*- coding: utf-8 -*-
"""Gradio Blocks 布局示例 - 基础的左右分栏交互界面

功能特性:
    - 使用 gr.Blocks() 自定义布局，而非 gr.Interface
    - 左右分栏：左侧为输入区（文本框 + 提交按钮），右侧为输出区
    - 简单的文本回显功能

依赖:
    - gradio

运行方式:
    - 直接运行: python block_api_demo1.py
"""
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():  # 水平行
        with gr.Column():  # 左侧列
            input_text = gr.Textbox(label="输入")
            submit_btn = gr.Button("提交")
        with gr.Column():  # 右侧列
            output_text = gr.Textbox(label="输出")
    submit_btn.click(fn=lambda x: f"你输入了: {x}", inputs=input_text, outputs=output_text)

demo.launch()
