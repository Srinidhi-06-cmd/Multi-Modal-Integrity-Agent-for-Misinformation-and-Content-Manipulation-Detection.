# src/models/train_fake_news_nn.py

import os
import re
import numpy as np
import pandas as pd
import nltk
import gensim
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random
import pickle

# ======================================================
# Reproducibility
# ======================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ======================================================
# Setup Paths
# ======================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(BASE_DIR, "data", "fake_news")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# Download Stopwords
# ======================================================
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ======================================================
# Load Dataset
# ======================================================
print("Loading dataset...")

fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
true = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]].dropna()

print("Dataset Loaded:", df.shape)

# ======================================================
# Text Cleaning
# ======================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return words

print("Cleaning text...")
df["clean_text"] = df["text"].apply(clean_text)

# ======================================================
# Train Word2Vec
# ======================================================
print("Training Word2Vec model...")

w2v_model = gensim.models.Word2Vec(
    sentences=df["clean_text"].tolist(),
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10,
    seed=SEED
)

w2v_model.save(os.path.join(MODEL_DIR, "fake_news_w2v.model"))
print("Word2Vec model saved.")

# ======================================================
# Convert Sentences to Embeddings
# ======================================================
def sentence_to_vec(sentence, model, vector_size=100):
    vec = np.zeros(vector_size)
    count = 0
    for word in sentence:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count != 0:
        vec /= count
    return vec

print("Converting text to vectors...")

X = np.array([sentence_to_vec(s, w2v_model) for s in df["clean_text"]])
y = df["label"].values

print("Feature Shape:", X.shape)

# ======================================================
# Train/Test Split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# ======================================================
# Build Neural Network
# ======================================================
def build_model(input_dim, optimizer_name="adam"):

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=0.001)
    elif optimizer_name == "sgd":
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=0.001)
    else:
        optimizer = Adam()

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ======================================================
# Early Stopping
# ======================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ======================================================
# Train With Multiple Optimizers
# ======================================================
optimizers = ["adam", "sgd", "rmsprop"]
results_list = []
history_dict = {}

for opt in optimizers:

    print("\n" + "="*50)
    print(f"Training with {opt.upper()} optimizer")
    print("="*50)

    model = build_model(X_train.shape[1], optimizer_name=opt)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    history_dict[opt] = history.history

    # Evaluation
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{opt.upper()} Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    np.save(os.path.join(MODEL_DIR, f"fake_news_cm_{opt}.npy"), cm)

    # Save ROC Data
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    np.save(os.path.join(MODEL_DIR, f"fake_news_fpr_{opt}.npy"), fpr)
    np.save(os.path.join(MODEL_DIR, f"fake_news_tpr_{opt}.npy"), tpr)

    # Save Model
    model.save(os.path.join(MODEL_DIR, f"fake_news_nn_{opt}.h5"))

    # Store for proof CSV
    results_list.append({
        "optimizer": opt.upper(),
        "best_train_accuracy": max(history.history["accuracy"]),
        "best_validation_accuracy": max(history.history["val_accuracy"]),
        "test_accuracy": acc
    })

# ======================================================
# Save Proof Files
# ======================================================

results_df = pd.DataFrame(results_list)

# Save CSV (for Streamlit display)
results_df.to_csv(os.path.join(MODEL_DIR, "fake_news_training_results.csv"), index=False)

# Save PKL (optional)
with open(os.path.join(MODEL_DIR, "fake_news_history.pkl"), "wb") as f:
    pickle.dump(history_dict, f)

# ======================================================
# Final Summary
# ======================================================
print("\n" + "="*50)
print("Optimizer Comparison Results")
print("="*50)
print(results_df)

best_model = results_df.sort_values(by="test_accuracy", ascending=False).iloc[0]

print(f"\nBest Optimizer: {best_model['optimizer']} "
      f"({best_model['test_accuracy']*100:.2f}%)")

print("\nAll models trained and saved successfully!")

