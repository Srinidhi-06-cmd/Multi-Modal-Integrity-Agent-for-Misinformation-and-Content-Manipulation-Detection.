import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ----------------------------
# Fix imports from src folder
# ----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "src"))

from models.predict_fake_news_nn import predict_news
from models.predict_dark_pattern_nn import predict_dark_pattern
from models.predict_synthetic_speech import predict_synthetic_audio
from models.predict_casia_cnn import predict_image  # ✅ Use function instead

# =====================================================
# Page Setup
# =====================================================
st.set_page_config(page_title="Integrity Agent", layout="wide")

st.title("🛡 Integrity Agent")
st.subheader("Multi-Modal AI: Fake News, Dark Patterns, Synthetic Speech & Deepfake Detection")

# =====================================================
# Model Training Proof
# =====================================================
st.markdown("## 📊 Model Training Proof")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📰 Fake News Model")
    fake_path = os.path.join(BASE_DIR, "saved_models", "fake_news_training_results.csv")
    if os.path.exists(fake_path):
        fake_df = pd.read_csv(fake_path)
        st.dataframe(fake_df, use_container_width=True)
    else:
        st.warning("Fake News training proof not found.")

with col2:
    st.markdown("### 🛒 Dark Pattern Model")
    dark_path = os.path.join(BASE_DIR, "saved_models", "dark_pattern_training_results.csv")
    if os.path.exists(dark_path):
        dark_df = pd.read_csv(dark_path)
        st.dataframe(dark_df, use_container_width=True)
    else:
        st.warning("Dark Pattern training proof not found.")

st.markdown("---")

# =====================================================
# Inputs
# =====================================================
news_input = st.text_area("📰 News Text")
text_input = st.text_area("🛒 Website Text")
audio_file = st.file_uploader("🎧 Upload Audio (.wav/.mp3)", type=["wav","mp3"])
image_file = st.file_uploader("🖼 Upload Image (.jpg/.png/.jpeg)", type=["jpg","png","jpeg"])

# =====================================================
# Button
# =====================================================
if st.button("🚀 Check Integrity"):

    scores = []

    # ---------------- Fake News ----------------
    if news_input.strip():
        st.markdown("### 📰 Fake News Result")
        prediction, fake_prob, real_prob = predict_news(news_input)

        if prediction == 1:
            st.error("⚠ Fake News Detected")
        else:
            st.success("✅ Real News")

        st.write(f"Fake Probability: {fake_prob*100:.2f}%")
        st.write(f"Real Probability: {real_prob*100:.2f}%")

        scores.append(real_prob)

    # ---------------- Dark Pattern ----------------
    if text_input.strip():
        st.markdown("### 🛒 Dark Pattern Result")
        result = predict_dark_pattern(text_input)

        if result == "Dark Pattern Detected":
            st.warning("⚠ Dark Pattern Detected")
            scores.append(0)
        else:
            st.success("✅ No Dark Pattern")
            scores.append(1)

    # ---------------- Synthetic Speech ----------------
    if audio_file:
        st.markdown("### 🎧 Synthetic Speech Result")

        temp_path = os.path.join(BASE_DIR, "temp_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(audio_file.getbuffer())

        result, fake_prob, real_prob = predict_synthetic_audio(temp_path)

        if result == "FAKE":
            st.error("⚠ Synthetic Voice Detected")
        else:
            st.success("✅ Real Human Voice")

        st.write(f"Real Probability: {real_prob*100:.2f}%")
        scores.append(real_prob)

    # ---------------- Deepfake Image ----------------
    if image_file:
        st.markdown("### 🖼 Deepfake Image Result")

        temp_img_path = os.path.join(BASE_DIR, "temp_image.jpg")
        with open(temp_img_path, "wb") as f:
            f.write(image_file.getbuffer())

        label, fake_prob, real_prob = predict_image(temp_img_path)

        if label == "FAKE":
            st.error("⚠ Fake / Manipulated Image Detected")
        else:
            st.success("✅ Real Image")

        st.write(f"Fake Probability: {fake_prob*100:.2f}%")
        st.write(f"Real Probability: {real_prob*100:.2f}%")

        scores.append(real_prob)

    # ---------------- Combined Score ----------------
    if scores:
        combined = sum(scores) / len(scores) * 100
        st.markdown("---")
        st.success(f"💡 Combined Integrity Score: {combined:.2f}/100")
    else:
        st.warning("Please provide at least one input.")