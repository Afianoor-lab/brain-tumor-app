import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from utils import preprocess_image, make_prediction, generate_pdf_report
import gdown
import os

# Download the model from Google Drive if not already present
MODEL_PATH = "brain_tumor_resnet50v2_hho_optimized.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "1HrCGlQi3ViiyfAbPaS7JOhZhujLzWBZ0"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

@st.cache_resource
def load_model_hho():
    return load_model(MODEL_PATH)

model = load_model_hho()

st.title("🧠 Brain Tumor Detection & Classification")
uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.info("Analyzing image...")
        image_array = preprocess_image(image)
        result = make_prediction(model, image_array)

        st.success("✅ Prediction Complete!")
        st.markdown(f"**Tumor Detected:** {'Yes' if result['has_tumor'] else 'No'}")
        if result['has_tumor']:
            st.markdown(f"**Tumor Type:** {result['tumor_type'].title()}")

        report_file = generate_pdf_report(image, result)
        with open(report_file, "rb") as f:
            st.download_button("📄 Download Report", f, file_name="report.pdf")
