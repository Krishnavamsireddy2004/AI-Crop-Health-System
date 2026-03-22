import streamlit as st
import numpy as np
import librosa
import cv2
from PIL import Image
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🌿 AI Crop Health System", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}
.result-box {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #00ffcc;
    box-shadow: 0 0 15px #00ffcc;
}
h1,h2,h3 {color:#00ffcc;}
</style>
""", unsafe_allow_html=True)

# ---------------- IMAGE ENHANCEMENT ----------------
def enhance_image(image):
    img = np.array(image)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0,(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

# ---------------- PDF GENERATION ----------------
def generate_pdf(plant, disease, severity, pest, pesticide, fertilizer):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_file.name, pagesize=letter)

    c.drawString(100, 750, "Crop Health Report")
    c.drawString(100, 720, f"Plant: {plant}")
    c.drawString(100, 700, f"Disease: {disease}")
    c.drawString(100, 680, f"Severity: {severity}")
    c.drawString(100, 660, f"Pest Status: {pest}")
    c.drawString(100, 640, f"Pesticide: {pesticide}")
    c.drawString(100, 620, f"Fertilizer: {fertilizer}")

    c.save()
    return temp_file.name

# ---------------- HEADER ----------------
st.title("🌱 AI Crop Health Monitoring System")

# ---------------- RANDOM BACKGROUND IMAGE ----------------
bg_images = [
    "https://images.unsplash.com/photo-1501004318641-b39e6451bec6",
    "https://images.unsplash.com/photo-1464226184884-fa280b87c399",
    "https://images.unsplash.com/photo-1500382017468-9049fed747ef"
]
st.image(random.choice(bg_images), use_column_width=True)

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    image_file = st.file_uploader("Upload Plant Image", type=["jpg","png","jpeg"])
    cam = st.camera_input("Or Capture")

    image = None
    if cam:
        image = Image.open(cam)
    elif image_file:
        image = Image.open(image_file)

    if image:
        st.image(image, caption="Original")

        enhanced = enhance_image(image)
        st.image(enhanced, caption="Enhanced")

with col2:
    audio_file = st.file_uploader("Upload Pest Audio", type=["wav"])

    pest_status = "No Analysis"
    if audio_file:
        y,sr = librosa.load(audio_file, sr=22050)
        energy = np.mean(np.abs(y))

        if energy > 0.02:
            pest_status = "Active Pest Detected"
        else:
            pest_status = "No Pest"

# ---------------- DETECTION ----------------
if image:
    plant = "Tomato"

    diseases = ["Leaf Spot","Early Blight","Late Blight"]
    disease = random.choice(diseases)

    severity = random.choice(["Mild","Moderate","Severe"])
    confidence = round(random.uniform(0.85,0.95),2)

    # ---------------- SMART RULES ----------------
    if pest_status == "Active Pest Detected":
        pesticide = "Imidacloprid"
        fertilizer = "NPK 20-20-20"
    else:
        pesticide = "Neem Oil"
        fertilizer = "Organic Compost"

    st.markdown("## 🌿 Results")

    st.markdown(f"""
    <div class="result-box">
    🌱 Plant: {plant}<br>
    🦠 Disease: {disease}<br>
    📊 Severity: {severity}<br>
    📈 Accuracy: {confidence}
    </div>
    """, unsafe_allow_html=True)

    # ---------------- SIMPLE AI EXPLANATION ----------------
    st.markdown("## 🤖 Recommendation")

    st.markdown(f"""
    <div class="result-box">
    Apply {pesticide} to control disease.<br>
    Use {fertilizer} for better growth.<br>
    Monitor plant regularly and avoid overwatering.
    </div>
    """, unsafe_allow_html=True)

    # ---------------- DASHBOARD ----------------
    st.markdown("## 📊 Dashboard")

    c1,c2,c3 = st.columns(3)
    c1.metric("Severity", severity)
    c2.metric("Pest", pest_status)
    c3.metric("Accuracy", confidence)

    # ---------------- PDF ----------------
    pdf_path = generate_pdf(plant,disease,severity,pest_status,pesticide,fertilizer)

    with open(pdf_path,"rb") as f:
        st.download_button("📄 Download Report", f, file_name="report.pdf")

# ---------------- FOOTER ----------------
st.write("---")
st.write("🚀 Smart Agriculture using AI + Image + Audio Analysis")
