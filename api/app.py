import os
import cv2
import numpy as np
import joblib
import urllib.request
import mediapipe as mp

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
from tensorflow.keras.models import load_model

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ============================================================
# PATH SETUP
# app.py ada di   : bahasa_isyarat/api/app.py
# models ada di   : bahasa_isyarat/models/
# hand_landmarker : bahasa_isyarat/hand_landmarker.task
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../api/
ROOT_DIR = os.path.dirname(BASE_DIR)                    # .../bahasa_isyarat/

MODEL_RF_PATH      = os.path.join(ROOT_DIR, "models", "best_rf_model.joblib")
DYNAMIC_MODEL_PATH = os.path.join(ROOT_DIR, "models", "dynamic_lstm.keras")
DYNAMIC_LABELS_PATH= os.path.join(ROOT_DIR, "models", "dynamic_labels.npy")
HAND_LANDMARKER    = os.path.join(ROOT_DIR, "hand_landmarker.task")
HAND_LANDMARKER_URL= (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# ============================================================
# FASTAPI
# ============================================================
app = FastAPI(title="Bahasa Isyarat AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DOWNLOAD hand_landmarker.task jika belum ada
# ============================================================
if not os.path.exists(HAND_LANDMARKER):
    print(f"[INFO] Mengunduh hand_landmarker.task ...")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER)
    print("[INFO] Download selesai.")

# ============================================================
# LOAD STATIC MODEL (RF / XGBoost) — huruf & angka
# ============================================================
loaded = joblib.load(MODEL_RF_PATH)

if isinstance(loaded, dict) and "model" in loaded and "label_encoder" in loaded:
    static_model = loaded["model"]
    static_le    = loaded["label_encoder"]
    print("[INFO] Static model (encoded) loaded.")
else:
    static_model = loaded
    static_le    = None
    print("[INFO] Static model (raw) loaded.")

# ============================================================
# LOAD DYNAMIC MODEL (LSTM) — kata
# ============================================================
dynamic_model  = None
dynamic_labels = None

if os.path.exists(DYNAMIC_MODEL_PATH) and os.path.exists(DYNAMIC_LABELS_PATH):
    dynamic_model  = load_model(DYNAMIC_MODEL_PATH)
    dynamic_labels = np.load(DYNAMIC_LABELS_PATH, allow_pickle=True)
    print(f"[INFO] Dynamic model loaded. Classes: {list(dynamic_labels)}")
else:
    print("[WARNING] Dynamic model tidak ditemukan.")

# ============================================================
# MEDIAPIPE TASKS — Hand Landmarker (SAMA dengan realtime_static.py)
# Pakai IMAGE mode karena kita kirim satu frame per request
# ============================================================
base_options = mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER)

options_static = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,   # IMAGE mode untuk per-frame
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = mp_vision.HandLandmarker.create_from_options(options_static)

# ============================================================
# SEQUENCE BUFFER — untuk LSTM (per session)
# ============================================================
SEQUENCE_LENGTH = 30
sequence_buffers: dict[str, deque] = {}

# ============================================================
# FEATURE EXTRACTION — STATIC
# IDENTIK dengan extract_features() di realtime_static.py:
#   Layout  : Kiri [0:63], Kanan [63:126]
#   Norm 1  : relatif ke pergelangan (landmark[0])
#   Norm 2  : bagi jarak maks ke pergelangan
# ============================================================
def extract_features_static(hand_landmarks_list, handedness_list) -> np.ndarray:
    features = np.zeros(126)

    for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
        if hand_idx >= 2:
            break

        label = handedness_list[hand_idx][0].category_name  # 'Left' / 'Right'

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        )  # (21, 3)

        # Normalisasi 1: Translation
        landmarks -= landmarks[0].copy()

        # Normalisasi 2: Scale
        max_dist = np.max(np.linalg.norm(landmarks, axis=1))
        if max_dist > 0:
            landmarks /= max_dist

        flat = landmarks.flatten()  # (63,)

        if label == "Left":
            features[0:63] = flat
        elif label == "Right":
            features[63:126] = flat

    return features


# ============================================================
# FEATURE EXTRACTION — DYNAMIC
# IDENTIK dengan collect_dynamic.py: raw x,y,z tanpa normalisasi
# ============================================================
def extract_features_dynamic(hand_landmarks_list) -> np.ndarray:
    keypoints = np.zeros(126)

    for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
        if hand_idx >= 2:
            break

        landmarks = np.array(
            [lm.x for lm in hand_landmarks] +
            [lm.y for lm in hand_landmarks] +
            [lm.z for lm in hand_landmarks],
            dtype=np.float32
        )

        # Flatten per landmark (x0,y0,z0, x1,y1,z1, ...)
        flat = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32
        ).flatten()

        if hand_idx == 0:
            keypoints[0:63] = flat
        else:
            keypoints[63:126] = flat

    return keypoints


# ============================================================
# HELPER: decode gambar dari upload, flip, konversi ke mp.Image
# ============================================================
def decode_frame(contents: bytes):
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None
    # JANGAN flip di sini — browser sudah mirror video sebelum kirim
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return frame, mp_image


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {
        "message": "Bahasa Isyarat AI — aktif",
        "endpoints": ["/predict", "/predict-dynamic"]
    }



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediksi huruf / angka — satu frame, pakai RF/XGBoost."""
    contents = await file.read()
    frame, mp_image = decode_frame(contents)

    if mp_image is None:
        return {"prediction": "Error: gambar tidak valid", "confidence": 0.0}

    results = detector.detect(mp_image)

    if not results.hand_landmarks:
        return {"prediction": "Tidak ada tangan", "confidence": 0.0}

    features = extract_features_static(results.hand_landmarks, results.handedness)

    pred_raw = static_model.predict([features])[0]

    confidence = 0.0
    if hasattr(static_model, "predict_proba"):
        proba      = static_model.predict_proba([features])[0]
        confidence = float(np.max(proba))

    predicted = static_le.inverse_transform([pred_raw])[0] if static_le else str(pred_raw)

    return {
        "prediction": str(predicted),
        "confidence": round(confidence * 100, 1),
    }


@app.post("/predict-dynamic")
async def predict_dynamic(
    file: UploadFile = File(...),
    session_id: str = "default",
):
    """Prediksi kata — buffer 30 frame per session, pakai LSTM."""
    if dynamic_model is None:
        return {"prediction": "Model dynamic tidak tersedia", "confidence": 0.0, "buffer_len": 0}

    contents = await file.read()
    frame, mp_image = decode_frame(contents)

    if mp_image is None:
        return {"prediction": "Error: gambar tidak valid", "confidence": 0.0, "buffer_len": 0}

    results = detector.detect(mp_image)

    hand_list = results.hand_landmarks if results.hand_landmarks else []
    keypoints = extract_features_dynamic(hand_list)

    if session_id not in sequence_buffers:
        sequence_buffers[session_id] = deque(maxlen=SEQUENCE_LENGTH)

    buf = sequence_buffers[session_id]
    buf.append(keypoints)

    if len(buf) < SEQUENCE_LENGTH:
        return {"prediction": None, "confidence": 0.0, "buffer_len": len(buf)}

    input_data = np.expand_dims(list(buf), axis=0)   # (1, 30, 126)
    prediction = dynamic_model.predict(input_data, verbose=0)[0]
    pred_class = int(np.argmax(prediction))
    confidence = float(prediction[pred_class])
    pred_word  = str(dynamic_labels[pred_class])

    return {
        "prediction": pred_word,
        "confidence": round(confidence * 100, 1),
        "buffer_len": SEQUENCE_LENGTH,
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sequence_buffers:
        del sequence_buffers[session_id]
    return {"message": f"Session '{session_id}' dihapus"}
