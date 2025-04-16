import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torch.nn.functional import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def build_image_index(image_paths):
    """
    输入图像路径列表，输出 (图像路径列表, 图像特征张量)
    """
    features = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_feat = clip_model.get_image_features(**inputs)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        features.append(image_feat.squeeze(0))
    feature_tensor = torch.stack(features, dim=0)  # [N, dim]
    return image_paths, feature_tensor


def retrieve_top_k_images(text_query, image_paths, image_tensor, top_k=3):
    if len(image_paths) == 0 or image_tensor.numel() == 0:
        print("⚠️ 图库为空，无法检索图像。")
        return []

    # 使用 CLIP 处理文本，得到向量表示
    inputs = clip_processor(text=[text_query], return_tensors="pt").to(device)
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**inputs)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # 对图像特征归一化
    image_tensor = image_tensor / image_tensor.norm(dim=-1, keepdim=True)

    # 计算 cosine similarity，通常返回 shape 为 [1, N]
    similarities = cosine_similarity(text_feat, image_tensor)

    # 如果 similarities 为二维 (如 [1, N])，则取第 0 行
    if similarities.dim() == 2:
        similarities = similarities[0]

    # 如果 similarities 为标量或0维，则扩展为1维张量
    if similarities.dim() == 0:
        similarities = similarities.unsqueeze(0)

    # 打印调试信息
    print(f"[DEBUG] 文本向量 shape: {text_feat.shape}")
    print(f"[DEBUG] 图像特征 shape: {image_tensor.shape}")
    print(f"[DEBUG] 相似度向量 shape: {similarities.shape}")

    # 确保 k 不超过可用元素个数
    k = min(top_k, similarities.numel())
    if k == 0:
        print("⚠️ 没有可用图像用于匹配。")
        return []

    topk = torch.topk(similarities, k=k)
    return [image_paths[i] for i in topk.indices.tolist()]


