import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONFIG
# =========================
DATASET_PATH = "dataset/dynamic"

SEQUENCE_LENGTH = 30
FEATURE_LENGTH = 126

MODEL_SAVE_PATH = "models/dynamic_lstm.keras"

# =========================
# LOAD DATASET
# =========================
sequences = []
labels = []

print("Loading dataset...")

# Ambil semua folder label
actions = sorted(os.listdir(DATASET_PATH))

for action in actions:

    action_path = os.path.join(DATASET_PATH, action)

    if not os.path.isdir(action_path):
        continue

    print(f"Loading class: {action}")

    # Ambil semua sequence
    for sequence in os.listdir(action_path):

        sequence_path = os.path.join(action_path, sequence)

        if not os.path.isdir(sequence_path):
            continue

        window = []

        # Ambil 30 frame
        for frame_num in range(SEQUENCE_LENGTH):

            frame_path = os.path.join(
                sequence_path,
                f"{frame_num}.npy"
            )

            if not os.path.exists(frame_path):
                break

            res = np.load(frame_path)

            window.append(res)

        # Pastikan lengkap 30 frame
        if len(window) == SEQUENCE_LENGTH:

            sequences.append(window)

            labels.append(action)

# =========================
# CONVERT TO ARRAY
# =========================
X = np.array(sequences)

print("Shape X :", X.shape)

# =========================
# LABEL ENCODER
# =========================
le = LabelEncoder()

y = le.fit_transform(labels)

y = to_categorical(y)

print("Shape y :", y.shape)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data :", X_train.shape)
print("Validation data :", X_val.shape)

# =========================
# BUILD MODEL
# =========================
model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        activation='relu',
        input_shape=(SEQUENCE_LENGTH, FEATURE_LENGTH)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        128,
        return_sequences=True,
        activation='relu'
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        64,
        return_sequences=False,
        activation='relu'
    )
)

model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(len(actions), activation='softmax'))

# =========================
# COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# CALLBACK
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop]
)

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)

model.save(MODEL_SAVE_PATH)

print(f"\nModel berhasil disimpan:")
print(MODEL_SAVE_PATH)

# =========================
# SAVE LABELS
# =========================
np.save(
    "models/dynamic_labels.npy",
    le.classes_
)

print("Label classes berhasil disimpan")

# =========================
# EVALUATION
# =========================
loss, accuracy = model.evaluate(X_val, y_val)

print(f"\nValidation Accuracy : {accuracy*100:.2f}%")