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
    try:
        prompt = f"""
        You are an agricultural expert.

        Plant: {plant}
        Disease: {disease}
        Severity: {severity}
        Pest Status: {pest}

        Provide:
        1. Disease explanation
        2. Recommended pesticide name
        3. Fertilizer name
        4. Prevention tips
        """

        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.7}
        )

        return response.text

    except Exception as e:
        return "⚠️ AI service temporarily unavailable. Please try again."

# ---------------- IMAGE ENHANCEMENT ----------------
def enhance_image(image):
    img = np.array(image)

    # Convert to LAB for contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return enhanced_img

# ---------------- UI ----------------
st.set_page_config(page_title="🌿 AI Crop Health System", layout="wide")

st.title("🌱 AI-Based Crop Health Monitoring System")
st.write("Upload plant image and pest audio to detect disease & get smart solutions")

# ---------------- IMAGE INPUT ----------------
st.subheader("📷 Upload or Capture Plant Image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Capture Image")

image = None

if camera_image:
    image = Image.open(camera_image)
elif uploaded_file:
    image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Original Image", use_column_width=True)

    # 🔥 Image Enhancement
    enhanced_img = enhance_image(image)
    st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)

    # ---------------- FAKE AI DETECTION ----------------
    diseases = ["Early Blight", "Late Blight", "Leaf Spot", "Healthy"]
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

pest_status = "No Data"
pesticide = "N/A"
fertilizer = "N/A"

if audio_file:
    y, sr = librosa.load(audio_file, sr=22050)
    energy = np.mean(np.abs(y))

    if energy > 0.02:
        pest_status = "Active Pest Detected"
        pesticide = "Chlorpyrifos"
        fertilizer = "NPK 20-20-20"
    else:
        pest_status = "No Pest Detected"
        pesticide = "Not Required"
        fertilizer = "Organic Compost"

    st.write("Pest Status:", pest_status)

# ---------------- GEMINI AI ----------------
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

col1.metric("🌿 Disease Severity", severity if image else "N/A")
col2.metric("🐛 Pest Activity", pest_status)
col3.metric("📈 Confidence", str(confidence) if image else "N/A")

# ---------------- EXTRA INFO ----------------
st.subheader("🌾 Recommended Treatment")

st.write("🧪 Pesticide:", pesticide)
st.write("🌱 Fertilizer:", fertilizer)

# ---------------- FOOTER ----------------
st.write("---")
st.write("🚀 Developed using Generative AI + IoT Concept")
