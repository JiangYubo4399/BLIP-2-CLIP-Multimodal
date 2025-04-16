import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from models.memory_prompt import MemoryManager


class BLIP2Wrapper:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BLIP2Wrapper] Loading model on {self.device}...")

        self.processor = Blip2Processor.from_pretrained(model_name)

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",  # 自动将模型切成多个部分加载到多GPU
            load_in_8bit=True,  # 使用8-bit量化加载，节省显存
            torch_dtype=torch.float16  # 强烈推荐开启半精度（更快且更省）
        )
        self.model.eval()

    def generate_caption(self, image: Image.Image, max_new_tokens: int = 50) -> str:
        """
        Generate a caption from an image using BLIP-2.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption

    def answer_question(self, image: Image.Image, question: str, max_new_tokens: int = 100) -> str:
        """
        Answer a question based on an image using BLIP-2.
        """
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return answer
    def answer_with_history(self, image, memory: MemoryManager, question: str):
        prompt = memory.get_prompt(question)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        memory.append(question, answer)
        return answer
