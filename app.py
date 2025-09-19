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

# Add background logo (PowerClip style)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{data.encode('base64').decode()}");
            background-size: 300px;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
            opacity: 0.1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call background (replace with your actual logo file path)
if os.path.exists("logo.png"):  # <-- ensure you have your logo saved as logo.png
    import base64
    with open("logo.png", "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded_logo}") no-repeat center;
            background-size: 350px;
            background-attachment: fixed;
            opacity: 0.15;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("🧬 APPLICATION FOR DETECTING HEPATITIS B VIRUS")
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

    if C == 1:  # grayscale model
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (W, H))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
    else:  # RGB model
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

        if file_type == "dcm":
            dicom = pydicom.dcmread(uploaded_file)
            img_array = dicom.pixel_array
            st.image(img_array, caption="Uploaded DICOM Image", use_container_width=True, clamp=True)
        else:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            st.image(img_array, caption="Uploaded Image", use_container_width=True, clamp=True)

        x = preprocess_image(img_array, H, W, C)
        label, prob, raw = safe_predict(model, x, threshold=threshold)

        st.write("🔎 Debug Info:")
        st.write("- Processed shape:", x.shape)
        st.write("- Pixel range:", x.min(), "to", x.max())
        st.write("- Raw model output:", raw)

        st.write(f"### 🩺 Prediction: **{label}**")
        st.write(f"Confidence: **{prob:.4f}** (Threshold = {threshold})")

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
    This HBV Detection App was developed as part of a research project to apply 
    **deep learning (CNN models)** for the detection of **Hepatitis B Virus (HBV)** from 
    medical imaging data.  

    ✅ Key Features:  
    - Upload and analyze **DICOM or common image formats (PNG, JPG, JPEG)**  
    - Automatic preprocessing of medical images (resizing, normalization, grayscale conversion)  
    - Configurable **prediction threshold** for classification tuning  
    - Save results with custom notes for research or clinical documentation  
    - Review past predictions in a built-in **gallery system**  

    ⚠️ Disclaimer:  
    This tool is intended **for research and educational purposes only**.  
    It is **not certified for clinical or diagnostic use**. Always consult a licensed medical professional 
    for health-related decisions.  
    """)            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        return img

    # File uploader
    uploaded_file = st.file_uploader(
        "📂 Choose a DICOM or Image file",
        type=["dcm", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type == "dcm":
            dicom = pydicom.dcmread(uploaded_file)
            img_array = dicom.pixel_array
            st.image(img_array, caption="Uploaded DICOM Image", use_container_width=True, clamp=True)
        else:
            image = Image.open(uploaded_file).convert("L")
            img_array = np.array(image)
            st.image(img_array, caption="Uploaded Image", use_container_width=True, clamp=True)

        processed_img = preprocess_image(img_array)

        # Debugging
        st.write("🔎 Debug Info:")
        st.write("- Processed shape:", processed_img.shape)
        st.write("- Pixel range:", processed_img.min(), "to", processed_img.max())

        # Prediction
        prediction = model.predict(processed_img)
        prob = prediction[0][0] if prediction.ndim > 1 else prediction[0]

        st.write("- Raw model output:", prediction)

        result = "HBV Positive 🟥" if prob > 0.7 else "HBV Negative 🟩"

        st.write(f"### 🩺 Prediction: **{result}**")
        st.write(f"Confidence: **{prob:.4f}**")

# =========================
# ABOUT PAGE
# =========================
elif page == "About":
    st.title("ℹ️ About This App")
    st.markdown(
        """
        This **Hepatitis B Virus (HBV) Detection App** was developed as part of my 
        **MSc Project**. The app leverages a **Convolutional Neural Network (CNN)** 
        trained on medical imaging data to predict **HBV Positive** or **HBV Negative** 
        cases from uploaded DICOM and image files.

        ### Features
        - Accepts DICOM, PNG, JPG, JPEG formats.
        - Uses deep learning for binary classification.
        - Provides probability confidence scores.
        - User-friendly web interface built with Streamlit.

        ### Author
        **Tolorunju Adedeji**  
        MSc Computer Science Project (2025)

        ---
        🧬 Powered by TensorFlow | 🖥️ Built with Streamlit
        """
    )

# =========================
# FOOTER
# =========================
st.markdown(
    "<div class='footer'>© 2025 Tolorunju Adedeji | MSc Project</div>",
    unsafe_allow_html=True
)
