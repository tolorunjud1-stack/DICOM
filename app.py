# app.py# app.py
import os
import json
import uuid
import pydicom
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd
from datetime import datetime
import pytz

# ==============================
# CONFIG
# ==============================
BASE_DIR = r"C:\Users\tolorunjud1\Desktop\DICOM1"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "processed")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(
    page_title="MSLBP-CNN HBV Project",
    page_icon="🏠",
    layout="wide"
)

# ==============================
# DATE & TIME DISPLAY
# ==============================
tz = pytz.timezone("Africa/Lagos")
now = datetime.now(tz)
current_time = now.strftime("%A, %d %B %Y | %H:%M:%S %Z")
st.markdown(f"<p style='text-align:right; color:gray;'>{current_time}</p>", unsafe_allow_html=True)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
.header { font-size:28px; color:#1a1a1a; text-align:center; padding:15px; border-bottom:2px solid #ddd; }
.subheader { font-size:20px; color:#333; margin-top:10px; }
.footer { font-size:14px; color:gray; text-align:center; margin-top:30px; }
.pred-card { padding:14px; border:1px solid #eee; border-radius:14px; margin-bottom:10px; background-color:#f9f9f9; }
.note-card { border:1px solid #ddd; border-radius:12px; padding:12px; margin-bottom:15px; background-color:#fefefe; }
</style>
""", unsafe_allow_html=True)

# ==============================
# MODEL LOADER
# ==============================
@st.cache_resource
def load_mslbp_model(path: str):
    try:
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.h5")
model, model_err = load_mslbp_model(MODEL_PATH)

def get_model_input_hwc(keras_model):
    shape = keras_model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[1]), int(shape[2]), int(shape[3])

# ==============================
# IMAGE/DICOM HELPERS
# ==============================
def dicom_to_uint8(ds):
    arr = ds.pixel_array
    try: arr = apply_voi_lut(arr, ds)
    except: pass
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = np.max(arr) - arr
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr.astype(np.float32) * slope + intercept
    arr = ((arr - np.min(arr)) / max(np.ptp(arr), 1e-5) * 255).astype(np.uint8)
    return arr

def load_regular_image(file, channels=1):
    img = Image.open(file)
    img = img.convert("L") if channels==1 else img.convert("RGB")
    return np.array(img)

def prepare_for_model(img, H, W, C):
    if C==1:
        if img.ndim==3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (W,H))
        x = img.astype(np.float32)/255.0
        x = np.expand_dims(x,-1)
    else:
        if img.ndim==2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (W,H))
        x = img.astype(np.float32)/255.0
    return np.expand_dims(x,0)

def safe_predict(model, x):
    y = model.predict(x)
    y = np.array(y)
    return float(y.flatten()[0])

def save_result_preview_and_json(img_disp, result, base_name):
    png_path = os.path.join(RESULTS_DIR, f"{base_name}.png")
    json_path = os.path.join(RESULTS_DIR, f"{base_name}.json")
    cv2.imwrite(png_path, img_disp if img_disp.ndim==2 else cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR))
    with open(json_path, "w") as f:
        json.dump(result,f,indent=2)
    return png_path,json_path

def load_results():
    results = []
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(".json"):
            try:
                with open(os.path.join(RESULTS_DIR,file)) as f:
                    results.append(json.load(f))
            except:
                continue
    return results

# ==============================
# SIDEBAR NAVIGATION
# ==============================
menu = st.sidebar.radio("Navigation", ["Home", "Gallery", "About"])

# ==============================
# HOME PAGE
# ==============================
if menu=="Home":
    st.markdown("<div class='header'><b>APPLICATION FOR DETECTING HEPATITIS B VIRUS USING MSLBP-CNN<b/></div>", unsafe_allow_html=True)
    if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, width=120)
    st.markdown("<div class='subheader'>Welcome to the MSLBP-CNN HBV Detection System</div>", unsafe_allow_html=True)
    st.write("Upload **DICOM** or **Image** files for HBV detection.")

    if model is None:
        st.error(f"Model not loaded from: {MODEL_PATH}")
        st.code(model_err or "Unknown error")
    else:
        H,W,C = get_model_input_hwc(model)
        st.caption(f"Model input shape: H={H}, W={W}, C={C}")

    uploaded_files = st.file_uploader("Upload files", type=["dcm","png","jpg","jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_id = str(uuid.uuid4())
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path,"wb") as f: f.write(file.read())
            try:
                if file.name.lower().endswith(".dcm"):
                    ds = pydicom.dcmread(file_path)
                    img_uint8 = dicom_to_uint8(ds)
                    metadata = {
                        "Patient ID": str(getattr(ds,"PatientID","N/A")),
                        "Patient Name": str(getattr(ds,"PatientName","N/A")),
                        "Modality": str(getattr(ds,"Modality","N/A")),
                        "Study Date": str(getattr(ds,"StudyDate","N/A")),
                        "Rows": str(getattr(ds,"Rows","N/A")),
                        "Columns": str(getattr(ds,"Columns","N/A"))
                    }
                    st.subheader(f"📋 Metadata - {file.name}")
                    st.json(metadata)
                    st.image(img_uint8, caption=f"DICOM Preview - {file.name}", use_container_width=True)
                else:
                    img_uint8 = load_regular_image(file_path, channels=C)
                    metadata = {}
                    st.subheader(f"🖼 Image Preview - {file.name}")
                    st.image(img_uint8, caption=file.name, use_container_width=True)

                x = prepare_for_model(img_uint8,H,W,C)
                prob = safe_predict(model,x)
                label = "HBV Positive" if prob>=0.5 else "HBV Negative"

                # Prediction card
                st.markdown(
                    f"<div class='pred-card'><b>🔬 Prediction:</b> {label}<br><b>Confidence:</b> {prob:.4f}</div>", 
                    unsafe_allow_html=True
                )

                # Note & buttons card
                st.markdown("<div class='note-card'>", unsafe_allow_html=True)
                note = st.text_area("📝 Leave a note about this prediction", key=f"note_{file_id}", height=80)
                result_payload = {
                    "file_name": file.name,
                    "label": label,
                    "confidence": float(f"{prob:.6f}"),
                    "metadata": metadata,
                    "note": note
                }
                col1, col2 = st.columns([1,1])
                with col1:
                    if st.button(f"✅ Keep {file.name}", key=f"keep_{file_id}"):
                        save_result_preview_and_json(img_uint8, result_payload, file_id)
                        st.success(f"{file.name} saved to gallery with your note!")
                with col2:
                    if st.button(f"🗑 Delete {file.name}", key=f"delete_{file_id}"):
                        try:
                            if os.path.exists(file_path): os.remove(file_path)
                            st.warning(f"{file.name} deleted.")
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

    st.markdown("<div class='footer'>© 2025 Tolorunju Adedeji | MSc Project</div>", unsafe_allow_html=True)

# ==============================
# GALLERY PAGE
# ==============================
elif menu=="Gallery":
    st.markdown("<div class='header'>Gallery of Processed Files</div>", unsafe_allow_html=True)
    results = load_results()
    if not results:
        st.info("No processed files yet.")
    else:
        csv_rows = []
        for res in results:
            st.subheader(res.get("file_name", "Unknown"))
            st.write("**Prediction:**", res.get("label","N/A"))
            st.write("**Confidence:**", res.get("confidence","N/A"))
            if "note" in res and res["note"]: st.info(f"📝 Note: {res['note']}")
            if "metadata" in res and res["metadata"]: st.json(res["metadata"])
            png_path = os.path.join(RESULTS_DIR,res["file_name"].split(".")[0]+".png")
            if os.path.exists(png_path): st.image(png_path,use_container_width=True)
            st.markdown("---")
            # Prepare CSV
            row = {"file_name":res.get("file_name",""), "prediction":res.get("label",""), "confidence":res.get("confidence","")}
            metadata = res.get("metadata",{})
            for k,v in metadata.items(): row[k]=v
            row["note"] = res.get("note","")
            csv_rows.append(row)
        df = pd.DataFrame(csv_rows)
        st.download_button("📥 Download CSV of Gallery", df.to_csv(index=False).encode("utf-8"), file_name="gallery.csv")

# ==============================
# ==============================
# ABOUT PAGE (fixed & polished)
# ==============================
elif menu == "About":
    st.markdown("<div class='header'>About This Project</div>", unsafe_allow_html=True)

    # Top section with logo on the right
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown("""
### Kwara State University, Malete  
**MSc Project** — **MSLBP-CNN HBV Detection System**

#### Supervisors
- **Dr. R. S. Babatunde** — Supervisor  
- **Dr. A. N. Babatunde** — Co-Supervisor

#### Project Brief
This project develops an advanced deep learning framework (**MSLBP-CNN**) for **Hepatitis B Virus (HBV)** detection from medical images.  
We combine **Multi-Scale Local Binary Patterns (MSLBP)** for robust texture encoding with a **Convolutional Neural Network (CNN)** for classification, aiming to improve **sensitivity**, **accuracy**, and **training efficiency** for liver imaging–based diagnosis.
        """)

    with col_right:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.caption("Logo not found")

    st.markdown("""
### Dataset
- **Source/Type:** Anonymized **DICOM** CT liver scans  
- **Key DICOM handling:** VOI-LUT, Rescale Slope/Intercept, MONOCHROME1 inversion  
- **Preprocessing:** Min–max normalization, resize to **128×128** (grayscale)  
- **Augmentation:** Rotations, flips, slight intensity jitter  
- **Splits:** Train / Validation / Test

### Method Overview
1. **MSLBP** at multiple radii/scales → stacked texture maps  
2. **CNN** classifier → predicts **HBV Positive / HBV Negative**  
3. **Evaluation metrics:** Accuracy, **Sensitivity (Recall)**, ROC-AUC, **Training time**  
4. **Validation:** **k-fold cross-validation** for robustness

### Why MSLBP-CNN?
- **MSLBP** captures fine-grained liver textures that can be subtle in HBV-related pathology.  
- **CNN** learns hierarchical patterns and decision boundaries from the texture-enhanced inputs.  
- The hybrid improves **discriminative power** and can reduce **training time** versus CNN-only pipelines.

""")

    st.markdown("<div class='footer'>© 2025 Tolorunju Adedeji | MSc Project</div>", unsafe_allow_html=True)
