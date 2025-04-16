# app.py

import gradio as gr
from PIL import Image
import os

from models.blip2_wrapper import BLIP2Wrapper
from utils.retrieval import build_image_index, retrieve_top_k_images

# ==== 初始化模型 ====
blip2 = BLIP2Wrapper()

# ==== 图像索引准备 ====
gallery_dir = "assets/gallery"
gallery_paths = [os.path.join(gallery_dir, fname) for fname in os.listdir(gallery_dir) if
                 fname.endswith((".jpg", ".png"))]
gallery_ids, gallery_feats = build_image_index(gallery_paths)

print("加载图库数量：", len(gallery_ids))
print("🧾 实际图像路径数量：", len(gallery_paths))
print("📐 图像特征 shape：", gallery_feats.shape)

# ==== 主处理函数 ====
def multimodal_pipeline(uploaded_img: Image.Image, user_question: str):
    if uploaded_img is None:
        return "请上传图像。", "", [], ""

    # 1. 自动生成图像描述（用于展示）
    caption = blip2.generate_caption(uploaded_img)

    # 2. 回答用户问题
    answer = blip2.answer_question(uploaded_img, user_question)

    # 3. 用 caption → CLIP 提取文本向量 → 与图库图像匹配
    retrieved_imgs = retrieve_top_k_images(
        text_query=caption,
        image_paths=gallery_ids,
        image_tensor=gallery_feats,
        top_k=3
    )

    return caption, answer, retrieved_imgs, f"检索关键词：{caption}"



# ==== Gradio 界面 ====
with gr.Blocks(title="BLIP-2 多模态智能系统") as demo:
    gr.Markdown("##BLIP-2+CLIP 多模态图像描述 + 问答 + 图文检索系统")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传图片")
            question_input = gr.Textbox(label="你想问这张图什么？", placeholder="例如：这是什么动物？")

            submit_btn = gr.Button("运行")

        with gr.Column():
            caption_output = gr.Textbox(label="自动生成描述")
            answer_output = gr.Textbox(label="图像问答回答")
            gallery_output = gr.Gallery(label="基于描述的图文检索 Top-3", columns=3, object_fit="contain",
                                        height="auto")
            retrieval_info = gr.Textbox(label="检索关键词（Caption）")

    submit_btn.click(fn=multimodal_pipeline,
                     inputs=[image_input, question_input],
                     outputs=[caption_output, answer_output, gallery_output, retrieval_info])

demo.launch()
