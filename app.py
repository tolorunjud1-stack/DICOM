import streamlit as st
import pydicom
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import json
import datetime

# ==============================
# APP CONFIG
# ==============================
st.set_page_config(page_title="HBV Detection App", layout="wide")

st.title("🧬 HBV Detection App")
st.write("Upload a medical image (DICOM, PNG, JPG, or JPEG) to get a prediction.")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_cnn_model():
    model = load_model("cnn_model.h5")
    return model

model = load_cnn_model()
st.success("✅ Model loaded successfully!")
st.write("📌 Model input shape:", model.input_shape)

# Expected input shape
_, H, W, C = model.input_shape

# ==============================
# SIDEBAR CONFIG
# ==============================
st.sidebar.markdown("### ⚙️ Settings")
threshold = st.sidebar.slider("Prediction Threshold", 0.1, 0.9, 0.5, 0.05)

# ==============================
# PREPROCESS & PREDICT
# ==============================
def preprocess_image(image, H, W, C):
    img = np.array(image)

    # Convert to grayscale if model expects 1 channel
    if C == 1:
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (W, H))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
    else:  # 3-channel RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (W, H))
        img = img.astype("float32") / 255.0

    return np.expand_dims(img, axis=0)

def safe_predict(model, img_array, threshold=0.5):
    raw = model.predict(img_array)
    prob = float(raw.flatten()[0])
    label = "HBV Positive 🟥" if prob >= threshold else "HBV Negative 🟩"
    return label, prob, raw

# ==============================
# STORAGE CONFIG
# ==============================
SAVE_DIR = "predictions"
os.makedirs(SAVE_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(SAVE_DIR, "results.json")

def save_result(file_name, label, prob, raw, notes=""):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "file": file_name,
        "label": label,
        "confidence": prob,
        "raw_output": raw.tolist(),
        "notes": notes,
        "timestamp": timestamp
    }
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=4)
    return result

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []

# ==============================
# MAIN APP (Pages)
# ==============================
page = st.sidebar.radio("📑 Pages", ["Home", "Gallery", "About"])

# ------------------------------
# HOME PAGE
# ------------------------------
if page == "Home":
    uploaded_file = st.file_uploader("📂 Choose a DICOM or Image file", type=["dcm", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()

        # Handle DICOM
        if file_type == "dcm":
            dicom = pydicom.dcmread(uploaded_file)
            img_array = dicom.pixel_array
            st.image(img_array, caption="Uploaded DICOM Image", use_container_width=True, clamp=True)
        else:
            # Regular image
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            st.image(img_array, caption="Uploaded Image", use_container_width=True, clamp=True)

        # Preprocess & Predict
        x = preprocess_image(img_array, H, W, C)
        label, prob, raw = safe_predict(model, x, threshold=threshold)

        # Debug Info
        st.write("🔎 Debug Info:")
        st.write("- Processed shape:", x.shape)
        st.write("- Pixel range:", x.min(), "to", x.max())
        st.write("- Raw model output:", raw)

        # Results
        st.write(f"### 🩺 Prediction: **{label}**")
        st.write(f"Confidence: **{prob:.4f}** (Threshold = {threshold})")

        # Notes
        notes = st.text_area("📝 Add notes for this case (optional)")
        if st.button("💾 Save Result"):
            save_result(uploaded_file.name, label, prob, raw, notes)
            st.success("Result saved!")

# ------------------------------
# GALLERY PAGE
# ------------------------------
elif page == "Gallery":
    st.subheader("📂 Saved Predictions")
    results = load_results()
    if not results:
        st.info("No saved results yet.")
    else:
        for res in results[::-1]:
            st.markdown(f"**File:** {res['file']}")
            st.markdown(f"- 🩺 Prediction: {res['label']}")
            st.markdown(f"- Confidence: {res['confidence']:.4f}")
            st.markdown(f"- Timestamp: {res['timestamp']}")
            if res.get("notes"):
                st.markdown(f"- 📝 Notes: {res['notes']}")
            st.markdown("---")

# ------------------------------
# ABOUT PAGE
# ------------------------------
elif page == "About":
    st.subheader("ℹ️ About This App")
    st.write("""
    This app is designed for **HBV detection** using a trained CNN model.  
    - Supports **DICOM, PNG, JPG, JPEG** images  
    - Preprocesses images automatically  
    - Allows you to adjust **threshold** for classification  
    - Saves results with notes for later review  

    ⚠️ Disclaimer: This app is for research and educational purposes only.  
    It is **not a substitute** for professional medical advice or diagnosis.
    """)
