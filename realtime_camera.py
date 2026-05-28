import cv2
import mediapipe as mp
import numpy as np
import joblib
import logging
import os

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

HAND_LANDMARKER_MODEL = "hand_landmarker.task"

# ─── Koneksi antar landmark (untuk menggambar garis di tangan) ────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Ibu jari
    (0, 5), (5, 6), (6, 7), (7, 8),           # Telunjuk
    (0, 9), (9, 10), (10, 11), (11, 12),      # Jari tengah
    (0, 13), (13, 14), (14, 15), (15, 16),    # Jari manis
    (0, 17), (17, 18), (18, 19), (19, 20),    # Kelingking
    (5, 9), (9, 13), (13, 17), (0, 17),       # Telapak
]

def draw_landmarks_on_image(image, hand_landmarks_list):
    """Gambar titik dan garis landmark tangan pada frame kamera."""
    h, w = image.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        # Konversi koordinat normalized ke pixel
        points = {
            i: (int(lm.x * w), int(lm.y * h))
            for i, lm in enumerate(hand_landmarks)
        }
        # Gambar garis koneksi
        for start, end in HAND_CONNECTIONS:
            cv2.line(image, points[start], points[end], (0, 200, 100), 2)
        # Gambar titik landmark
        for pt in points.values():
            cv2.circle(image, pt, 4, (255, 255, 255), -1)
            cv2.circle(image, pt, 4, (0, 150, 80), 1)


def extract_features(hand_landmarks_list, handedness_list):
    """
    Ekstrak 126 fitur dari hasil deteksi MediaPipe Tasks API.
    Layout: Kiri [0:63], Kanan [63:126]
    """
    features = np.zeros(126)

    for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
        if hand_idx >= 2:
            break

        label = handedness_list[hand_idx][0].category_name  # 'Left' atau 'Right'

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        )  # (21, 3)

        # Normalisasi 1: Translation (relatif ke pergelangan)
        landmarks -= landmarks[0].copy()

        # Normalisasi 2: Scale (bagi dengan jarak maks ke pergelangan)
        max_dist = np.max(np.linalg.norm(landmarks, axis=1))
        if max_dist > 0:
            landmarks /= max_dist

        flat = landmarks.flatten()  # (63,)
        if label == "Left":
            features[0:63] = flat
        elif label == "Right":
            features[63:126] = flat

    return features


def main():
    # ── Load model ────────────────────────────────────────────────────────────
    model_path = "models/best_rf_model.joblib"
    try:
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict) and "model" in loaded and "label_encoder" in loaded:
            model = loaded["model"]
            le    = loaded["label_encoder"]
            is_encoded = True
            logging.info(f"Model XGBoost/Encoded berhasil dimuat dari {model_path}")
        else:
            model = loaded
            le    = None
            is_encoded = False
            logging.info(f"Model RF standar berhasil dimuat dari {model_path}")
    except FileNotFoundError:
        logging.error(f"File model tidak ditemukan: {model_path}. Latih model dulu.")
        return

    # ── Inisialisasi MediaPipe Tasks Hand Landmarker ──────────────────────────
    if not os.path.exists(HAND_LANDMARKER_MODEL):
        logging.error(
            f"File '{HAND_LANDMARKER_MODEL}' tidak ditemukan. "
            "Jalankan train_advanced.py sekali agar file ter-download otomatis, "
            "atau download manual dari:\n"
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        )
        return

    base_options = mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,   # mode VIDEO untuk kamera real-time
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    # ── Buka kamera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Gagal membuka kamera.")
        return

    logging.info("Kamera terbuka. Tekan 'q' untuk keluar.")

    frame_timestamp_ms = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logging.warning("Frame kamera kosong, dilewati.")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Bungkus ke format MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Deteksi — mode VIDEO butuh timestamp monoton (milidetik)
        frame_timestamp_ms += 33  # ≈ 30 fps
        results = detector.detect_for_video(mp_image, frame_timestamp_ms)

        hand_detected = bool(results.hand_landmarks)

        if hand_detected:
            draw_landmarks_on_image(frame, results.hand_landmarks)
            features = extract_features(results.hand_landmarks, results.handedness)

            pred_raw = model.predict([features])[0]
            predicted_class = le.inverse_transform([pred_raw])[0] if is_encoded else pred_raw

            # Tampilkan hasil prediksi
            cv2.putText(
                frame,
                f"Prediksi: {predicted_class}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Tidak ada tangan",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Real-time Sign Language Recognition", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()