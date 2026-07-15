# src/models/predict_casia_cnn.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==========================================================
# Paths
# ==========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "casia_mobilenet_fixed.keras")

# ==========================================================
# Load Model (Lazy Loading for Performance)
# ==========================================================
_model = None

def load_cnn_model():
    global _model
    if _model is None:
        print("Loading CASIA CNN model...")
        _model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    return _model

# ==========================================================
# Prediction Function
# ==========================================================
def predict_image(img_path, target_size=(224, 224)):
    """
    Predict whether image is REAL or FAKE.

    Returns:
        label (str): "REAL" or "FAKE"
        fake_prob (float)
        real_prob (float)
        confidence (float)
    """

    try:
        model = load_cnn_model()

        # ---------------- Load Image ----------------
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)

        # ---------------- Normalize ----------------
        x = x / 255.0

        # ---------------- Expand Dimensions ----------------
        x = np.expand_dims(x, axis=0)

        # ---------------- Predict ----------------
        pred = model.predict(x, verbose=0)

        # Convert to scalar
        pred_value = float(np.squeeze(pred))

        # ---------------- Probability Logic ----------------
        # Assuming model trained with:
        # 0 = FAKE, 1 = REAL
        real_prob = pred_value
        fake_prob = 1 - pred_value

        # ---------------- Label ----------------
        label = "REAL" if real_prob >= 0.5 else "FAKE"

        # ---------------- Confidence ----------------
        confidence = max(real_prob, fake_prob)

        return label, fake_prob, real_prob, confidence

    except Exception as e:
        print("Prediction Error:", str(e))
        return "ERROR", 0.0, 0.0, 0.0