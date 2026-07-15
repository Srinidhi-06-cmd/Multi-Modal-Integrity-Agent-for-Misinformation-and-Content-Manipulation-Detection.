import streamlit as st
import sys
import os
import pandas as pd

# ----------------------------
# Fix imports
# ----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "src"))

from models.predict_fake_news_nn import predict_news
from models.predict_dark_pattern_nn import predict_dark_pattern
from models.predict_casia_cnn import predict_image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Integrity Agent", layout="wide")

st.title("🛡 Integrity Agent")
st.markdown("### Multi-Modal AI for Detecting Digital Manipulation")
st.markdown("---")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🔍 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📰 Fake News Detection",
        "🛒 Dark Pattern Detection",
        "🖼 Deepfake Detection",
        "🧠 Full Integrity Check",
        "📊 Model Insights"
    ]
)

# =====================================================
# 🏠 HOME
# =====================================================
if page == "🏠 Home":
    st.header("Welcome to Integrity Agent")

    st.info("Detect fake news, dark patterns, and deepfake images using AI.")

    st.markdown("""
    ### 🔍 Features:
    - 📰 Fake News Detection  
    - 🛒 Dark Pattern Detection  
    - 🖼 Deepfake Image Detection  
    - 🧠 Combined Integrity Score  
    """)

# =====================================================
# 📰 FAKE NEWS
# =====================================================
elif page == "📰 Fake News Detection":
    st.header("📰 Fake News Detection")

    news_input = st.text_area("Enter News Article")

    if st.button("Analyze News"):
        if not news_input.strip():
            st.warning("Please enter news text")
        else:
            prediction, fake_prob, real_prob, confidence = predict_news(news_input)

            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.error("⚠ Fake News Detected")
                else:
                    st.success("✅ Real News")

                st.metric("Confidence", f"{confidence*100:.2f}%")
                st.progress(confidence)

            with col2:
                st.write(f"Fake Probability: {fake_prob*100:.2f}%")
                st.write(f"Real Probability: {real_prob*100:.2f}%")

                if confidence < 0.6:
                    st.warning("⚠ Low Confidence Prediction")

# =====================================================
# 🛒 DARK PATTERN
# =====================================================
elif page == "🛒 Dark Pattern Detection":
    st.header("🛒 Dark Pattern Detection")

    text_input = st.text_area("Enter Website Text")

    if st.button("Analyze Text"):
        if not text_input.strip():
            st.warning("Please enter text")
        else:
            result = predict_dark_pattern(text_input)

            if result == "Dark Pattern Detected":
                st.warning("⚠ Dark Pattern Detected")
            else:
                st.success("✅ No Dark Pattern")

# =====================================================
# 🖼 DEEPFAKE
# =====================================================
elif page == "🖼 Deepfake Detection":
    st.header("🖼 Deepfake Image Detection")

    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if image_file:
        col1, col2 = st.columns(2)

        with col1:
            st.image(image_file, caption="Uploaded Image", use_container_width=True)

        # Save temp image
        temp_path = os.path.join(BASE_DIR, "temp.jpg")
        with open(temp_path, "wb") as f:
            f.write(image_file.getbuffer())

        label, fake_prob, real_prob, confidence = predict_image(temp_path)

        with col2:
            if label == "FAKE":
                st.error("⚠ Fake / Manipulated Image")
            else:
                st.success("✅ Real Image")

            st.metric("Confidence", f"{confidence*100:.2f}%")
            st.progress(confidence)

            st.write(f"Fake Probability: {fake_prob*100:.2f}%")
            st.write(f"Real Probability: {real_prob*100:.2f}%")

            if confidence < 0.6:
                st.warning("⚠ Low Confidence Prediction")

# =====================================================
# 🧠 FULL SYSTEM
# =====================================================
elif page == "🧠 Full Integrity Check":
    st.header("🧠 Full Integrity Analysis")

    news_input = st.text_area("News Text")
    text_input = st.text_area("Website Text")
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if st.button("Run Analysis"):

        scores = []

        if news_input.strip():
            _, _, real_prob, _ = predict_news(news_input)
            scores.append(real_prob)

        if text_input.strip():
            result = predict_dark_pattern(text_input)
            scores.append(1 if result != "Dark Pattern Detected" else 0)

        if image_file:
            temp_path = os.path.join(BASE_DIR, "temp.jpg")
            with open(temp_path, "wb") as f:
                f.write(image_file.getbuffer())

            _, _, real_prob, _ = predict_image(temp_path)
            scores.append(real_prob)

        if scores:
            score = sum(scores) / len(scores)
            st.success(f"💡 Integrity Score: {score*100:.2f}/100")
            st.progress(score)
        else:
            st.warning("Provide at least one input")

# =====================================================
# 📊 MODEL INSIGHTS
# =====================================================
elif page == "📊 Model Insights":
    st.header("📊 Model Insights & Performance")

    col1, col2 = st.columns(2)

    fake_path = os.path.join(BASE_DIR, "saved_models", "fake_news_training_results.csv")
    dark_path = os.path.join(BASE_DIR, "saved_models", "dark_pattern_training_results.csv")

    with col1:
        st.subheader("📰 Fake News")
        if os.path.exists(fake_path):
            st.dataframe(pd.read_csv(fake_path), use_container_width=True)

    with col2:
        st.subheader("🛒 Dark Pattern")
        if os.path.exists(dark_path):
            st.dataframe(pd.read_csv(dark_path), use_container_width=True)

    st.markdown("---")

    # Confusion Matrices
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📰 Confusion Matrix")
        for opt in ["adam", "sgd", "rmsprop"]:
            path = os.path.join(BASE_DIR, "saved_models", f"fake_news_cm_{opt}.png")
            if os.path.exists(path):
                st.image(path, caption=opt.upper(), use_container_width=True)

    with col2:
        st.subheader("🛒 Confusion Matrix")
        for opt in ["adam", "sgd", "rmsprop"]:
            path = os.path.join(BASE_DIR, "saved_models", f"dark_pattern_cm_{opt}.png")
            if os.path.exists(path):
                st.image(path, caption=opt.upper(), use_container_width=True)

    st.markdown("---")

    # Architecture
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📰 Fake News Architecture")
        for opt in ["adam", "sgd", "rmsprop"]:
            path = os.path.join(BASE_DIR, f"saved_models/fake_news_model_summary_{opt}.txt")
            if os.path.exists(path):
                with st.expander(opt.upper()):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        st.code(f.read())

    with col2:
        st.subheader("🛒 Dark Pattern Architecture")
        for opt in ["adam", "sgd", "rmsprop"]:
            path = os.path.join(BASE_DIR, f"saved_models/dark_pattern_model_summary_{opt}.txt")
            if os.path.exists(path):
                with st.expander(opt.upper()):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        st.code(f.read())