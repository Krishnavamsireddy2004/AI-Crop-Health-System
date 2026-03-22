import streamlit as st
import numpy as np
import librosa
import cv2
from PIL import Image
import random
import google.generativeai as genai

# ---------------- GEMINI SETUP ----------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_ai_solution(plant, disease, severity, pest):
    prompt = f"""
    Give a detailed agricultural solution:
    Plant: {plant}
    Disease: {disease}
    Severity: {severity}
    Pest Status: {pest}

    Include:
    - Recommended pesticide
    - Fertilizer suggestion
    - Prevention tips
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ---------------- UI DESIGN ----------------
st.set_page_config(page_title="🌿 AI Crop Health System", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5fff5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌱 AI-Based Crop Health Monitoring System")
st.write("Upload plant image and pest audio to detect disease & get solutions")

# ---------------- IMAGE INPUT ----------------
st.subheader("📷 Upload Plant Image")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

camera_image = st.camera_input("Or Capture Image")

if camera_image:
    image = Image.open(camera_image)
elif uploaded_file:
    image = Image.open(uploaded_file)
else:
    image = None

if image:
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------------- FAKE AI DETECTION ----------------
    diseases = [
        "Tomato Early Blight",
        "Tomato Late Blight",
        "Leaf Spot",
        "Healthy"
    ]
    severity_levels = ["Mild", "Moderate", "Severe"]

    plant_name = "Tomato Leaf"
    disease_name = random.choice(diseases)
    severity = random.choice(severity_levels)
    confidence = round(random.uniform(0.80, 0.95), 2)

    st.subheader("🌿 Detection Results")
    st.write(f"Plant: {plant_name}")
    st.write(f"Disease: {disease_name}")
    st.write(f"Severity: {severity}")
    st.write(f"Confidence: {confidence}")

# ---------------- AUDIO INPUT ----------------
st.subheader("🎤 Upload Pest Sound")
audio_file = st.file_uploader("Upload Audio", type=["wav"])

pest_status = "Unknown"

if audio_file:
    y, sr = librosa.load(audio_file, sr=22050)
    energy = np.mean(np.abs(y))

    if energy > 0.02:
        pest_status = "Active Pest Detected"
    else:
        pest_status = "No Significant Pest Activity"

    st.write("Pest Status:", pest_status)

# ---------------- GEMINI AI OUTPUT ----------------
if image:
    st.subheader("🤖 AI Expert Recommendation")

    ai_result = get_ai_solution(
        plant_name,
        disease_name,
        severity,
        pest_status
    )

    st.success(ai_result)

# ---------------- DASHBOARD ----------------
st.subheader("📊 Farmer Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("🌿 Disease Risk", severity if image else "N/A")
col2.metric("🐛 Pest Activity", pest_status)
col3.metric("📈 Confidence", str(confidence) if image else "N/A")

# ---------------- FOOTER ----------------
st.write("---")
st.write("Developed using Generative AI + IoT Concept 🚀")
