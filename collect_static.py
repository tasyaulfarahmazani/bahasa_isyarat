import cv2
import os

# =========================
# INPUT LABEL
# =========================
label = input("Masukkan label (contoh: A / 0 / halo / makan): ").strip()

# Validasi kosong
if label == "":
    print("Label tidak boleh kosong")
    exit()

# =========================
# MODE DATASET
# =========================
mode = input("Mode dataset (train/val): ").lower()

if mode not in ["train", "val"]:
    print("Mode harus train atau val")
    exit()

# =========================
# FOLDER PENYIMPANAN
# =========================
save_dir = f"dataset/images/{mode}/{label}"

os.makedirs(save_dir, exist_ok=True)

# Hitung jumlah file yang sudah ada
counter = len(os.listdir(save_dir))

# =========================
# BUKA KAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera gagal dibuka")
    exit()

print("\n===== PETUNJUK =====")
print("Tekan S -> Simpan gambar")
print("Tekan Q -> Keluar")
print("=====================\n")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame gagal dibaca")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # =========================
    # TAMPILKAN INFO
    # =========================
    cv2.putText(
        frame,
        f"Label : {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Jumlah : {counter}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Tekan S untuk simpan",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Tekan Q untuk keluar",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Collect Static Dataset", frame)
 
    key = cv2.waitKey(1) & 0xFF

    # =========================
    # SIMPAN GAMBAR
    # =========================
    if key == ord('s'):

        filename = os.path.join(save_dir, f"{counter}.jpg")

        cv2.imwrite(filename, frame)

        print(f"[SAVED] {filename}")

        counter += 1

    # =========================
    # KELUAR
    # =========================
    elif key == ord('q'):
        break

# =========================
# RELEASE
# =========================
cap.release()
cv2.destroyAllWindows()