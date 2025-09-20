import streamlit as st
import pydicom
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

st.title("🧬 HBV Detection App (Debug Mode)")
st.write("Upload a medical image (DICOM, PNG, JPG, or JPEG) to get a prediction.")

# Load the trained model
@st.cache_resource
def load_cnn_model():
    model = load_model("cnn_model.h5")
    return model

model = load_cnn_model()
st.success("✅ Model loaded successfully!")
st.write("📌 Model input shape:", model.input_shape)

# Preprocess the image before prediction
def preprocess_image(image):
    # Ensure input is numpy
    img = np.array(image)

    # If already grayscale (2D), skip conversion
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to match model input (224x224)
    img = cv2.resize(img, (224, 224))

    # Normalize pixel values
    img = img.astype("float32") / 255.0

    # Expand dimensions -> (1, 224, 224, 1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img

# File uploader
uploaded_file = st.file_uploader("📂 Choose a DICOM or Image file", type=["dcm", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    # Handle DICOM files
    if file_type == "dcm":
        dicom = pydicom.dcmread(uploaded_file)
        img_array = dicom.pixel_array
        st.image(img_array, caption="Uploaded DICOM Image", use_container_width=True, clamp=True)
    else:
        # Handle regular image files
        image = Image.open(uploaded_file).convert("L")  # force grayscale
        img_array = np.array(image)
        st.image(img_array, caption="Uploaded Image", use_container_width=True, clamp=True)

    # Preprocess and predict
    processed_img = preprocess_image(img_array)

    # Debugging info
    st.write("🔎 Debug Info:")
    st.write("- Processed shape:", processed_img.shape)
    st.write("- Pixel range:", processed_img.min(), "to", processed_img.max())

    # Prediction
    prediction = model.predict(processed_img)
    prob = prediction[0][0] if prediction.ndim > 1 else prediction[0]

    st.write("- Raw model output:", prediction)

    # Adjusted threshold
    result = "HBV Positive 🟥" if prob > 0.7 else "HBV Negative 🟩"

    st.write(f"### 🩺 Prediction: **{result}**")
    st.write(f"Confidence: **{prob:.4f}**")
