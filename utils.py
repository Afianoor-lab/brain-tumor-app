import numpy as np
from PIL import Image
from fpdf import FPDF
import datetime
import os

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def make_prediction(model, image_array):
    preds = model.predict(image_array)[0]
    has_tumor = np.argmax(preds[:2]) == 1
    tumor_classes = ['glioma', 'meningioma', 'pituitary']
    tumor_type = tumor_classes[np.argmax(preds[2:])] if has_tumor else "None"
    return {"has_tumor": has_tumor, "tumor_type": tumor_type}

def generate_pdf_report(image, result, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Brain Tumor Detection Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Tumor Detected: {'Yes' if result['has_tumor'] else 'No'}", ln=True)
    pdf.cell(200, 10, txt=f"Tumor Type: {result['tumor_type'].title()}", ln=True)
    image_path = "temp_img.jpg"
    image.save(image_path)
    pdf.image(image_path, x=10, y=70, w=100)
    pdf.output(filename)
    os.remove(image_path)
    return filename
