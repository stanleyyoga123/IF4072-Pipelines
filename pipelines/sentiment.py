import os
import numpy as np

from gensim.utils import simple_preprocess

from transformers import AutoTokenizer
from tensorflow.keras.models import load_model


class PipelineSentiment:
    def __init__(self):
        self.maxlen = 30
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = load_model(os.path.join("bin", "model"))

    def __call__(self, text):
        cleaned_text = " ".join(simple_preprocess(text))
        decoder = {
            1: "positive",
            2: "negative",
            0: "neutral",
        }
        tokens = self.tokenizer(
            [cleaned_text],
            max_length=self.maxlen,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        pred_proba = self.model.predict(tokens.data, batch_size=1)  # change this line
        sentiment = np.argmax(pred_proba[0])
        label = decoder[sentiment]
        confidence = pred_proba[0][sentiment]
        return label, confidence
