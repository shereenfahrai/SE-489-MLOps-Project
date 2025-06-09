import os

import joblib
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 拼接到 models 目录（位于 fake_news_detection/ 旁边）
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "models", "tokenizer.pkl")

# 加载模型和 tokenizer
model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)
