import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2TextGenerator:
    def __init__(self, max_length, temperature, top_k):
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k

        self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.eval()

    async def generate_text(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            max_length=self.max_length,
            temperature=self.temperature,
            top_k=self.top_k,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
