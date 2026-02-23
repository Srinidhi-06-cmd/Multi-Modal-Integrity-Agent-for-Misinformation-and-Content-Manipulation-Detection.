# src/models/train_dark_pattern.py

import os
import pandas as pd
import numpy as np
import re
import nltk
import gensim
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split

# -------------------------------
# Setup
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "dark_pattern", "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Load Dataset
# -------------------------------
print("\nLoading Dataset...\n")

df = pd.read_csv(DATA_PATH)
df = df[["text", "label"]]

print("Dataset Loaded:", df.shape)

# -------------------------------
# Clean Text
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

df["clean_text"] = df["text"].apply(clean_text)

# -------------------------------
# Train Word2Vec
# -------------------------------
print("\nTraining Word2Vec model...\n")

w2v_model = gensim.models.Word2Vec(
    sentences=df["clean_text"].tolist(),
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

w2v_model.save(os.path.join(MODEL_DIR, "dark_pattern_w2v.model"))

print("Word2Vec model saved successfully.")

# -------------------------------
# Convert Sentences to Vectors
# -------------------------------
def sentence_to_vec(sentence):
    vec = np.zeros(100)
    count = 0
    for word in sentence:
        if word in w2v_model.wv:
            vec += w2v_model.wv[word]
            count += 1
    return vec / count if count != 0 else vec

X = np.array([sentence_to_vec(s) for s in df["clean_text"]])
y = df["label"].values

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData split complete.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -------------------------------
# Build Feedforward Neural Network
# -------------------------------
def build_model(input_shape, optimizer_name="adam"):

    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    if optimizer_name == "adam":
        optimizer = Adam()
    elif optimizer_name == "sgd":
        optimizer = SGD()
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop()
    else:
        optimizer = Adam()

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# -------------------------------
# Train With Multiple Optimizers
# -------------------------------
results = []

for opt in ["adam", "sgd", "rmsprop"]:

    print(f"\n==============================")
    print(f"Training Dark Pattern NN with {opt.upper()} optimizer")
    print(f"==============================\n")

    model = build_model(X_train.shape[1], optimizer_name=opt)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n{opt.upper()} Test Accuracy: {acc:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"dark_pattern_nn_{opt}.h5")
    model.save(model_path)
    print(f"Model saved at: {model_path}")

    # Store results for proof
    results.append({
        "optimizer": opt.upper(),
        "best_train_accuracy": max(history.history["accuracy"]),
        "best_validation_accuracy": max(history.history["val_accuracy"]),
        "test_accuracy": acc
    })

# -------------------------------
# Save Proof File
# -------------------------------
results_df = pd.DataFrame(results)

results_path = os.path.join(MODEL_DIR, "dark_pattern_training_results.csv")
results_df.to_csv(results_path, index=False)

print("\nTraining Results Saved At:", results_path)

# -------------------------------
# Show Summary
# -------------------------------
print("\nTraining Summary:")
print(results_df)

best_model = results_df.sort_values(by="test_accuracy", ascending=False).iloc[0]

print("\nBest Optimizer Based on Test Accuracy:")
print(f"{best_model['optimizer']} "
      f"({best_model['test_accuracy']*100:.2f}% accuracy)")

print("\nAll Dark Pattern models trained and saved successfully!\n")
