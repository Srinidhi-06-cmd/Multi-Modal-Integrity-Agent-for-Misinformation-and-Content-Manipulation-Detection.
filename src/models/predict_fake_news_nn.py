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
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "fake_news_nn_adam.h5")
W2V_PATH = os.path.join(BASE_DIR, "saved_models", "fake_news_w2v.model")

model = load_model(MODEL_PATH)
w2v_model = gensim.models.Word2Vec.load(W2V_PATH)

# ----------------------------
# Cleaning
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

# ----------------------------
# Sentence → Vector
# ----------------------------
def sentence_to_vec(sentence):
    vec = np.zeros(100)
    count = 0

    for word in sentence:
        if word in w2v_model.wv:
            vec += w2v_model.wv[word]
            count += 1

    if count != 0:
        vec /= count

    return vec

# ----------------------------
# Prediction
# ----------------------------
def predict_news(text):
    words = clean_text(text)
    vec = sentence_to_vec(words).reshape(1, -1)

    prob = model.predict(vec, verbose=0)[0][0]

    fake_prob = float(prob)
    real_prob = float(1 - prob)

    prediction = 1 if fake_prob > 0.5 else 0
    confidence = max(fake_prob, real_prob)

    return prediction, fake_prob, real_prob, confidence