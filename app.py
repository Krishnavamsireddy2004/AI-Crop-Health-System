import streamlit as st
import numpy as np
import librosa
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import google.generativeai as genai

# ---------- GEMINI SETUP ----------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------- PAGE ----------
st.set_page_config(page_title="AI Crop Health", layout="centered")
st.title("🌿 AI Crop Health Analyzer")

# ---------- LOAD MODELS ----------
image_model = load_model("plant_model.h5")
audio_model = load_model("audio_model.h5")

# ---------- CLASS NAMES ----------
class_names = [
'Pepper__bell___Bacterial_spot','Pepper__bell___healthy',
'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight',
'Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
'Tomato_Spider_mites_Two_spotted_spider_mite',
'Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus',
'Tomato__Tomato_mosaic_virus','Tomato_healthy'
]

# ---------- GEMINI FUNCTION ----------
def get_ai_solution(plant, disease, severity, pest):
    try:
        prompt = f"""
        A plant named {plant} is affected by {disease}.
        Severity level is {severity}.
        Pest activity: {pest}.

        Provide:
        - Simple explanation
        - Best pesticide name
        - Best fertilizer name
        - Prevention tips
        """

        response = model.generate_content(prompt)
        return response.text
    except:
        return "⚠️ AI recommendation not available"

# ---------- INPUT ----------
st.subheader("📷 Upload or Capture Image")
image_file = st.file_uploader("Upload Leaf Image", type=["jpg","png"])
camera_image = st.camera_input("Use Camera")

st.subheader("🎤 Pest Audio (Optional)")
audio_file = st.file_uploader("Upload Audio", type=["wav"])

# ---------- IMAGE ENHANCEMENT ----------
def enhance_image(image):
    img = np.array(image)
    img = cv2.GaussianBlur(img, (5,5), 0)
    return Image.fromarray(img)

# ---------- IMAGE PREDICTION ----------
def predict_image(image):
    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = image_model.predict(img_array)
    return pred

# ---------- AUDIO ----------
def predict_audio(file):
    y, sr = librosa.load(file, sr=22050)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec = librosa.power_to_db(spec)
    spec = np.expand_dims(spec, axis=(0,-1))
    pred = audio_model.predict(spec)
    return pred

# ---------- MAIN ----------
image_input = camera_image if camera_image else image_file

if image_input:
    image = Image.open(image_input)

    enhanced = enhance_image(image)
    st.image(enhanced, caption="Enhanced Image")

    pred = predict_image(enhanced)
    confidence = np.max(pred)
    pred_class = np.argmax(pred)

    if confidence < 0.5:
        st.error("❌ Please upload a valid plant leaf image")
        st.stop()

    disease_full = class_names[pred_class]

    # Extract plant & disease
    plant_name = disease_full.split("_")[0]
    disease_name = disease_full.split("___")[-1]

    # Severity
    if confidence > 0.8:
        severity = "Severe"
    elif confidence > 0.6:
        severity = "Moderate"
    else:
        severity = "Mild"

    # Audio
    pest_status = "Not Detected"
    if audio_file:
        audio_pred = predict_audio(audio_file)
        pest_status = "Detected" if np.argmax(audio_pred)==1 else "Not Detected"

    # Final Result
    if severity == "Severe" and pest_status == "Detected":
        final = "HIGH RISK"
    elif severity == "Moderate":
        final = "MODERATE RISK"
    else:
        final = "LOW RISK"

    # Basic Solution
    if severity == "Mild":
        solution = "Use organic neem spray"
    elif severity == "Moderate":
        solution = "Apply fungicide"
    else:
        solution = "Use strong pesticide immediately"

    fertilizer = "Use compost" if pest_status == "Not Detected" else "Use pest-control fertilizer"

    # ---------- OUTPUT ----------
    st.subheader("📊 Results")

    st.write("🌱 Plant:", plant_name)
    st.write("🌿 Disease:", disease_name)
    st.write("📊 Severity:", severity)
    st.write("🐛 Pest:", pest_status)
    st.write("🔥 Final:", final)
    st.write("💊 Solution:", solution)
    st.write("🌾 Fertilizer:", fertilizer)

    # ---------- GEMINI OUTPUT ----------
    ai_result = get_ai_solution(plant_name, disease_name, severity, pest_status)

    st.subheader("🤖 AI Expert Recommendation")
    st.success(ai_result)
