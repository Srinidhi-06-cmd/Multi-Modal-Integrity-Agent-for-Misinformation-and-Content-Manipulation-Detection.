import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model


# -----------------------------
# MFCC Feature Extraction
# -----------------------------
def extract_mfcc(audio_path, n_mfcc=40, max_len=130):
    """
    Extract MFCC features from audio file.
    Output shape: (40, 130)
    """

    try:
        # IMPORTANT: Must match training sample rate
        y, sr = librosa.load(audio_path, sr=22050)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Pad or Trim to fixed length
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(
                mfcc,
                pad_width=((0, 0), (0, pad_width)),
                mode="constant"
            )
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc

    except Exception as e:
        print("❌ Error extracting MFCC:", e)
        return None


# -----------------------------
# Prediction Function
# -----------------------------
def predict_synthetic_audio(audio_path):
    """
    Returns:
        result: "REAL" or "FAKE"
        fake_prob: probability of fake
        real_prob: probability of real
    """

    model_path = "saved_models/synthetic_speech_model.h5"

    if not os.path.exists(model_path):
        return "MODEL_NOT_FOUND", 0.0, 0.0

    if not os.path.exists(audio_path):
        return "AUDIO_NOT_FOUND", 0.0, 0.0

    # Load trained CNN model
    model = load_model(model_path)

    # Extract MFCC features
    mfcc = extract_mfcc(audio_path)

    if mfcc is None:
        return "ERROR", 0.0, 0.0

    # Reshape for CNN input
    mfcc = mfcc.reshape(1, 40, 130, 1)

    # Predict (Softmax output → [real_prob, fake_prob])
    pred = model.predict(mfcc, verbose=0)[0]

    real_prob = float(pred[0])   # Class 0 = REAL
    fake_prob = float(pred[1])   # Class 1 = FAKE

    # Determine predicted class
    class_index = np.argmax(pred)

    if class_index == 1:
        return "FAKE", fake_prob, real_prob
    else:
        return "REAL", fake_prob, real_prob


# -----------------------------
# CLI Running (Optional)
# -----------------------------
if __name__ == "__main__":
    audio_path = input("Enter audio file path (.wav/.mp3): ").strip()

    result, fake_prob, real_prob = predict_synthetic_audio(audio_path)

    print("\n==============================")
    print("🎧 Synthetic Speech Detection Result")
    print("==============================")

    if result == "FAKE":
        print("⚠️ FAKE / SYNTHETIC SPEECH DETECTED")
    elif result == "REAL":
        print("✅ REAL / HUMAN SPEECH DETECTED")
    else:
        print("❌ ERROR:", result)

    print(f"\nFake Probability: {fake_prob*100:.2f}%")
    print(f"Real Probability: {real_prob*100:.2f}%")
    print("==============================\n")
