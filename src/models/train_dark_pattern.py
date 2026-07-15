import os
import pandas as pd
import numpy as np
import re
import nltk
import gensim
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import random

# -------------------------------
# Reproducibility
# -------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

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
df = df[["text", "label"]].dropna()

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
print("\nTraining Word2Vec...\n")

w2v_model = gensim.models.Word2Vec(
    sentences=df["clean_text"].tolist(),
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=10,
    seed=SEED
)

w2v_model.save(os.path.join(MODEL_DIR, "dark_pattern_w2v.model"))

# -------------------------------
# Convert to vectors
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
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# -------------------------------
# Model
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
        optimizer = Adam(learning_rate=0.001)
    elif optimizer_name == "sgd":
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
    else:
        optimizer = RMSprop(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# -------------------------------
# Early Stopping 🔥
# -------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -------------------------------
# Train Models
# -------------------------------
results = []

for opt in ["adam", "sgd", "rmsprop"]:

    print(f"\nTraining with {opt.upper()}...\n")

    model = build_model(X_train.shape[1], opt)

    # Save architecture
    with open(os.path.join(MODEL_DIR, f"dark_pattern_model_summary_{opt}.txt"), "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Prediction
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = np.mean(y_pred.flatten() == y_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{opt.upper()} Confusion Matrix")
    plt.savefig(os.path.join(MODEL_DIR, f"dark_pattern_cm_{opt}.png"))
    plt.close()

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(MODEL_DIR, f"dark_pattern_metrics_{opt}.csv")
    )

    # Save model
    model.save(os.path.join(MODEL_DIR, f"dark_pattern_nn_{opt}.h5"))

    results.append({
        "optimizer": opt.upper(),
        "accuracy": acc
    })

# -------------------------------
# Save results
# -------------------------------
pd.DataFrame(results).to_csv(
    os.path.join(MODEL_DIR, "dark_pattern_training_results.csv"),
    index=False
)

print("\n✅ Dark Pattern Training Completed!")