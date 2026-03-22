import streamlit as st
from PIL import Image
import numpy as np

st.title("🌿 AI Crop Health Detection System")

image_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])
audio_file = st.file_uploader("Upload Pest Audio (Optional)", type=["wav"])

if image_file:
    image = Image.open(image_file)
    st.image(image)

    diseases = ["Tomato Early Blight", "Healthy", "Leaf Spot"]
    disease = np.random.choice(diseases)

    severity = np.random.choice(["Mild", "Moderate", "Severe"])

    st.write("🌿 Disease:", disease)
    st.write("📊 Severity:", severity)

    pest = "Not Detected"
    if audio_file:
        pest = np.random.choice(["Detected", "Not Detected"])

    st.write("🐛 Pest Status:", pest)

    if severity == "Severe" and pest == "Detected":
        final = "🚨 HIGH RISK"
    elif severity == "Moderate":
        final = "⚠️ MODERATE RISK"
    else:
        final = "✅ LOW RISK"

    st.write("🔥 Final Result:", final)

    if severity == "Mild":
        solution = "Use organic spray"
    elif severity == "Moderate":
        solution = "Apply fungicide"
    else:
        solution = "Use strong pesticide immediately"

    st.write("💊 Solution:", solution)
    import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("gemini-pro")
