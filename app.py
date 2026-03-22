import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🌿 AI Crop Health", layout="wide")

# ---------------- BACKGROUND IMAGE ----------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
    background-size: cover;
    background-attachment: fixed;
}
.result-box {
    background: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 15px;
    color: white;
    border: 1px solid #00ffcc;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# ---------------- CLASS LABELS ----------------
class_names = {
    0: 'Pepper Bacterial Spot',
    1: 'Pepper Healthy',
    2: 'Potato Early Blight',
    3: 'Potato Late Blight',
    4: 'Potato Healthy',
    5: 'Tomato Bacterial Spot',
    6: 'Tomato Early Blight',
    7: 'Tomato Late Blight',
    8: 'Tomato Leaf Mold',
    9: 'Tomato Septoria',
    10: 'Tomato Spider Mites',
    11: 'Tomato Target Spot',
    12: 'Tomato Yellow Curl Virus',
    13: 'Tomato Mosaic Virus',
    14: 'Tomato Healthy'
}

# ---------------- IMAGE PROCESS ----------------
def preprocess_image(image):
    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UI ----------------
st.title("🌱 AI Crop Health Monitoring System")

image_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image")

    # ---------------- PREDICTION ----------------
    processed = preprocess_image(image)
    pred = model.predict(processed)

    class_index = np.argmax(pred)
    confidence = round(np.max(pred),2)

    label = class_names[class_index]

    # ---------------- EXTRACT INFO ----------------
    if "Tomato" in label:
        plant = "Tomato"
    elif "Potato" in label:
        plant = "Potato"
    elif "Pepper" in label:
        plant = "Pepper"
    else:
        plant = "Unknown"

    if "Healthy" in label:
        disease = "Healthy"
        severity = "None"
        pesticide = "Not Required"
        fertilizer = "Organic Compost"
    else:
        disease = label.replace(plant,"")
        severity = "Moderate" if confidence < 0.9 else "Severe"
        pesticide = "Imidacloprid"
        fertilizer = "NPK 20-20-20"

    # ---------------- OUTPUT ----------------
    st.markdown(f"""
    <div class="result-box">
    🌿 Plant: {plant} <br>
    🦠 Disease: {disease} <br>
    📊 Severity: {severity} <br>
    📈 Accuracy: {confidence}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-box">
    🧪 Pesticide: {pesticide} <br>
    🌱 Fertilizer: {fertilizer}
    </div>
    """, unsafe_allow_html=True)
