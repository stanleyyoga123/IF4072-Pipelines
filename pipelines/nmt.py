import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


class PipelineNMT:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained(
            os.path.join("bin", "model-nmt")
        )
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

    def preprocess(self, text):
        text = text.lower()
        text = text.replace("[^ a-z.?!,¿]", "")
        text = text.replace("[.?!,¿]", r" \0 ")
        text = text.strip()
        text = f"translate Indonesian to English: {text} </s>"
        tokenized = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        return tokenized

    def __call__(self, text):
        input_ = self.preprocess(text)
        out = self.model.generate(input_)[0]
        out = self.tokenizer.decode(out[out > 1])
        return out
