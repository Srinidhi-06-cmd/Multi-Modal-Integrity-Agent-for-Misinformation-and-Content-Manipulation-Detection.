import os
import numpy as np
from src.models.preprocess_audio import extract_mfcc
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "data/synthetic_speech/deepfake_voice"
REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")

MODEL_SAVE_PATH = "saved_models/synthetic_speech_model.h5"


def load_dataset():
    X = []
    y = []

    print("Loading REAL samples...")
    for file in os.listdir(REAL_PATH):
        if file.endswith(".wav") or file.endswith(".mp3"):
            file_path = os.path.join(REAL_PATH, file)
            mfcc = extract_mfcc(file_path)

            if mfcc is not None:
                X.append(mfcc)
                y.append(0)  # REAL = 0

    print("Loading FAKE samples...")
    for file in os.listdir(FAKE_PATH):
        if file.endswith(".wav") or file.endswith(".mp3"):
            file_path = os.path.join(FAKE_PATH, file)
            mfcc = extract_mfcc(file_path)

            if mfcc is not None:
                X.append(mfcc)
                y.append(1)  # FAKE = 1

    X = np.array(X)
    y = np.array(y)

    print("Total Samples:", len(X))
    print("Real Samples:", sum(y == 0))
    print("Fake Samples:", sum(y == 1))

    # CNN requires 4D input
    X = X[..., np.newaxis]

    return X, y


def build_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset()

    print("Dataset loaded!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    y_cat = to_categorical(y, num_classes=2)

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_model(input_shape=X.shape[1:])

    print("Training model...")
    model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=1
    )

    print("Saving model...")
    model.save(MODEL_SAVE_PATH)

    print("Model saved successfully at:", MODEL_SAVE_PATH)
