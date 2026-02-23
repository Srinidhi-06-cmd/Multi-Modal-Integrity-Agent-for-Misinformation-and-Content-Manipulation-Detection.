import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------- Paths --------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
CASIA_MODEL_PATH = os.path.join(MODEL_DIR, "casia_mobilenet_colab_gpu.keras")

# Load Sequential model (your trained model)
model = load_model(CASIA_MODEL_PATH, compile=False)

# -------------------- Prediction Function --------------------
def predict_image(img_path, target_size=(224,224)):
    """
    Predict if an image is real or deepfake using the CASIA MobileNet model.
    """
    # Load and preprocess image
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # shape (1,H,W,C)
    x = x / 255.0  # normalize

    # Predict
    pred = model.predict(x)

    # Squeeze scalar
    pred = np.squeeze(pred)
    if pred > 0.5:
        return "REAL", float(pred), float(1 - pred)
    else:
        return "FAKE", float(1 - pred), float(pred)
