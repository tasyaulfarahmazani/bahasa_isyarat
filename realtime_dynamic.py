import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model("models/dynamic_lstm.keras")

labels = np.load(
    "models/dynamic_labels.npy",
    allow_pickle=True
)

# =========================
# CONFIG
# =========================
SEQUENCE_LENGTH = 30
THRESHOLD = 0.8

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# =========================
# SEQUENCE BUFFER
# =========================
sequence = deque(maxlen=SEQUENCE_LENGTH)

sentence = []

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera gagal dibuka")
    exit()

print("Realtime dynamic recognition berjalan...")
print("Tekan Q untuk keluar")

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # =========================
    # FEATURE VECTOR
    # =========================
    keypoints = np.zeros(126)

    if results.multi_hand_landmarks:

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            if hand_idx >= 2:
                break

            # Draw landmark
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)

            # tangan pertama
            if hand_idx == 0:
                keypoints[0:63] = landmarks

            # tangan kedua
            else:
                keypoints[63:126] = landmarks

    # =========================
    # APPEND SEQUENCE
    # =========================
    sequence.append(keypoints)

    # =========================
    # PREDICTION
    # =========================
    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.expand_dims(sequence, axis=0)

        prediction = model.predict(input_data, verbose=0)[0]

        predicted_class = np.argmax(prediction)

        confidence = prediction[predicted_class]

        predicted_word = labels[predicted_class]

        # =========================
        # CONFIDENCE FILTER
        # =========================
        if confidence > THRESHOLD:

            if len(sentence) == 0:
                sentence.append(predicted_word)

            elif predicted_word != sentence[-1]:
                sentence.append(predicted_word)

    # =========================
    # LIMIT HISTORY
    # =========================
    if len(sentence) > 5:
        sentence = sentence[-5:]

    # =========================
    # DISPLAY
    # =========================
    cv2.rectangle(frame, (0,0), (640,80), (0,0,0), -1)

    cv2.putText(
        frame,
        " ".join(sentence),
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2,
        cv2.LINE_AA
    )

    # Confidence
    if len(sequence) == SEQUENCE_LENGTH:

        cv2.putText(
            frame,
            f"{predicted_word} ({confidence*100:.1f}%)",
            (20,110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Realtime Dynamic Recognition", frame)

    # =========================
    # KEYBOARD
    # =========================
    key = cv2.waitKey(1) & 0xFF

    # Reset sentence
    if key == ord('c'):
        sentence = []

    # Quit
    elif key == ord('q'):
        break

# =========================
# RELEASE
# =========================
cap.release()
cv2.destroyAllWindows()