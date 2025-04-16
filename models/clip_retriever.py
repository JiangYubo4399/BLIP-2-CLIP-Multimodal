from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt").to("cuda")
    image_feat = clip_model.get_image_features(**inputs)
    return image_feat / image_feat.norm(dim=-1, keepdim=True)

def get_text_embedding(text: str):
    inputs = clip_processor(text=[text], return_tensors="pt").to("cuda")
    text_feat = clip_model.get_text_features(**inputs)
    return text_feat / text_feat.norm(dim=-1, keepdim=True)

def retrieve_similar_images(query: str, image_paths: list, image_feats: torch.Tensor, top_k=3):
    query_feat = get_text_embedding(query)
    sims = torch.nn.functional.cosine_similarity(query_feat, image_feats)
    indices = torch.topk(sims, k=top_k).indices
    return [image_paths[i] for i in indices]
