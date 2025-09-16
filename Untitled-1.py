# =============================
# 1. Install dependencies (run once in your Jupyter Notebook)
# =============================
!pip install pydicom opencv-python tqdm numpy

# =============================
# 2. Import libraries
# =============================
import os
import numpy as np
import pydicom
import cv2
from tqdm import tqdm

# =============================
# 3. Define input & output paths
# =============================
# Update this to your local DICOM dataset folder
dicom_root = r"C:\Users\USER\Desktop\DICOM\manifest\Colorectal-Liver-Metastases"

# Folder to save preprocessed 3D volumes (.npy)
save_dir = r"C:\Users\Desktop\DICOM\preprocessed_images"
os.makedirs(save_dir, exist_ok=True)

# =============================
# 4. Function: Load 3D DICOM series
# =============================
def load_dicom_series(series_path):
    """Load a series of DICOM slices into a 3D volume (numpy array)."""
    slices = []
    for f in os.listdir(series_path):
        if f.lower().endswith(".dcm"):
            try:
                dcm = pydicom.dcmread(os.path.join(series_path, f))
                if hasattr(dcm, "InstanceNumber"):
                    slices.append((dcm.InstanceNumber, dcm.pixel_array))
            except:
                pass

    # Sort slices by InstanceNumber (proper 3D order)
    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=0).astype(np.float32)
    return volume if len(slices) > 0 else None

# =============================
# 5. Preprocessing Loop (3D)
# =============================
processed_count = 0

for root, dirs, files in os.walk(dicom_root):
    if any(f.lower().endswith(".dcm") for f in files):
        try:
            volume = load_dicom_series(root)
            if volume is None:
                continue

            # Normalize volume to [0,255]
            volume = cv2.normalize(volume, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Resize slices to (224x224) for CNN input
            resized_volume = np.stack([cv2.resize(slice_, (224, 224)) for slice_ in volume])

            # Save as .npy (one volume per folder)
            save_name = os.path.basename(root) + ".npy"
            save_path = os.path.join(save_dir, save_name)
            np.save(save_path, resized_volume)

            processed_count += 1

        except Exception as e:
            print(f"❌ Error processing {root}: {e}")

print(f"\n✅ Total 3D volumes processed and saved: {processed_count}")
print(f"📂 Saved in: {save_dir}")
