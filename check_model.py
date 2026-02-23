import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "casia_mobilenet_colab_gpu.keras")

model = load_model(MODEL_PATH, compile=False)

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Last layer:", model.layers[-1])