import os
import numpy as np
import gensim
from tensorflow.keras.models import load_model
import re
import nltk
from nltk.corpus import stopwords

# ----------------------------
# Setup
# ----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "dark_pattern_nn_adam.h5")
W2V_PATH = os.path.join(BASE_DIR, "saved_models", "dark_pattern_w2v.model")

model = load_model(MODEL_PATH)
w2v_model = gensim.models.Word2Vec.load(W2V_PATH)

# ----------------------------
# Text Cleaning
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

# ----------------------------
# Sentence Vector
# ----------------------------
def sentence_to_vec(sentence):
    vec = np.zeros(100)
    count = 0
    for word in sentence:
        if word in w2v_model.wv:
            vec += w2v_model.wv[word]
            count += 1
    return vec / count if count != 0 else vec

# ----------------------------
# 🚀 PREDICTION FUNCTION (IMPROVED)
# ----------------------------
def predict_dark_pattern(text):

    # ---------------- CLEAN + VECTOR ----------------
    words = clean_text(text)
    vec = sentence_to_vec(words).reshape(1, -1)

    # ---------------- MODEL PREDICTION ----------------
    prob = model.predict(vec)[0][0]

    # 🔥 Assume model output = probability of CLEAN (based on your issue)
    clean_prob = float(prob)
    dark_prob = float(1 - prob)

    # ---------------- RULE-BASED DETECTION ----------------
    text_lower = text.lower()

    keywords = [
        "only today", "limited time", "hurry", "last chance",
        "almost sold out", "few units", "don’t miss out",
        "offer ends", "act fast", "buy now", "exclusive deal",
        "before it's gone", "selling fast"
    ]

    rule_flag = any(k in text_lower for k in keywords)

    # ---------------- FINAL DECISION ----------------
    if rule_flag:
        label = "Dark Pattern Detected"

    elif dark_prob > 0.7:
        label = "Dark Pattern Detected"

    elif dark_prob < 0.3:
        label = "No Dark Pattern"

    else:
        label = "Uncertain"

    # ---------------- CONFIDENCE ----------------
    confidence = max(dark_prob, clean_prob)

    return label, dark_prob, clean_prob, confidence