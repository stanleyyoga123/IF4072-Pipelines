import os
import re
import pickle
import numpy as np

from gensim.models import Word2Vec
import tensorflow as tf

import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize


class PipelineLangDetect:
    def __init__(self):
        LANG_DETECT_MODELS_PATH = os.path.join("bin", "lang-detection")
        lang_encoder_path = os.path.join(
            LANG_DETECT_MODELS_PATH, "label_encoder.pickle"
        )
        lang_w2v_path = os.path.join(LANG_DETECT_MODELS_PATH, "word2vec_wiki_12.model")
        lang_model_path = os.path.join(LANG_DETECT_MODELS_PATH, "ann_word2vec_12.h5")

        with open(lang_encoder_path, "rb") as handle:
            self.lang_encoder = pickle.load(handle)
        self.w2v_lang_model = Word2Vec.load(lang_w2v_path)
        self.lang_model = tf.keras.models.load_model(lang_model_path)

    def regex_filter(self, text):
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9.†]', " ", text)
        text = re.sub(r"[[]]", " ", text)
        text = text.lower()
        return text

    def word_vector(self, tokens, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in tokens:
            try:
                vec += self.w2v_lang_model.wv[word].reshape((1, size))
                count += 1
            except KeyError:  # handling the case where the token is not in vocabulary
                continue
        if count != 0:
            vec /= count
        else:
            print("WARNING: all OOV")
        return vec

    def preprocess(self, sentence):
        clean_sent = self.regex_filter(sentence)
        return self.word_vector(word_tokenize(clean_sent), 200)

    def __call__(self, text):
        x = self.preprocess(text)
        lang = self.lang_model.predict(x)
        pred = np.zeros_like(lang)
        pred[np.arange(len(lang)), lang.argmax(1)] = 1
        pred_lang = self.lang_encoder.inverse_transform(pred)
        lang_code = pred_lang[0]
        if lang_code not in ["eng", "ind"]:
            return lang_code, False
        return lang_code, True
