class MemoryManager:
    def __init__(self):
        self.history = []

    def append(self, question, answer):
        self.history.append((question, answer))

    def get_prompt(self, current_question):
        prompt = ""
        for q, a in self.history[-3:]:  # 只取近 3 轮
            prompt += f"Question: {q}\nAnswer: {a}\n"
        prompt += f"Question: {current_question}\nAnswer:"
        return prompt
