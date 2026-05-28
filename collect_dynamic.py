import cv2
import os
import numpy as np
import mediapipe as mp

# =========================
# INPUT LABEL
# =========================
label = input("Masukkan kata: ").strip()

if label == "":
    print("Label tidak boleh kosong")
    exit()

# =========================
# CONFIG
# =========================
SEQUENCE_LENGTH = 30
NUM_SEQUENCES = 30

SAVE_PATH = f"dataset/dynamic/{label}"

os.makedirs(SAVE_PATH, exist_ok=True)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera gagal dibuka")
    exit()

print("\n===== PETUNJUK =====")
print("SPACE  -> mulai rekam sequence")
print("Q      -> keluar")
print("==============================\n")

sequence = 0

# =========================
# MAIN LOOP
# =========================
while sequence < NUM_SEQUENCES:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    # =========================
    # TAMPILAN AWAL
    # =========================
    cv2.putText(
        frame,
        f"Kata : {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Sequence : {sequence}/{NUM_SEQUENCES}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        "Tekan SPACE untuk mulai",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "Tekan Q untuk keluar",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.imshow("Collect Dynamic Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # KELUAR
    # =========================
    if key == ord('q'):
        break

    # =========================
    # MULAI RECORD
    # =========================
    elif key == 32:  # SPACE

        sequence_path = os.path.join(
            SAVE_PATH,
            f"sequence_{sequence}"
        )

        os.makedirs(sequence_path, exist_ok=True)

        print(f"\nRecording sequence {sequence}...")

        # =========================
        # RECORD 30 FRAME
        # =========================
        for frame_num in range(SEQUENCE_LENGTH):

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

                    landmarks = []

                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    landmarks = np.array(landmarks)

                    if hand_idx == 0:
                        keypoints[0:63] = landmarks
                    else:
                        keypoints[63:126] = landmarks

            # =========================
            # SAVE NPY
            # =========================
            np.save(
                os.path.join(sequence_path, f"{frame_num}.npy"),
                keypoints
            )

            # =========================
            # DISPLAY RECORDING
            # =========================
            cv2.putText(
                frame,
                f"RECORDING {label}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

            cv2.putText(
                frame,
                f"Frame : {frame_num}/{SEQUENCE_LENGTH}",
                (20,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
            )

            cv2.imshow("Collect Dynamic Dataset", frame)

            cv2.waitKey(1)

        print(f"Sequence {sequence} selesai disimpan")

        sequence += 1

cap.release()
cv2.destroyAllWindows()