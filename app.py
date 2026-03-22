from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# ---------------- HUGGINGFACE API ----------------
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
headers = {"Authorization": "Bearer YOUR_HF_API_KEY"}

def predict_image(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response.json()

# ---------------- SMART LOGIC ----------------
def analyze_result(label):
    label = label.lower()

    if "tomato" in label:
        plant = "Tomato"
    elif "potato" in label:
        plant = "Potato"
    elif "pepper" in label:
        plant = "Pepper"
    else:
        plant = "Unknown Plant"

    if "leaf" in label or "spot" in label:
        disease = "Leaf Disease"
        severity = "Moderate"
        pesticide = "Neem Oil"
        fertilizer = "NPK 20-20-20"
    else:
        disease = "Healthy"
        severity = "Low"
        pesticide = "Not Required"
        fertilizer = "Organic Compost"

    return plant, disease, severity, pesticide, fertilizer

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_bytes = file.read()

            prediction = predict_image(image_bytes)

            if isinstance(prediction, list):
                label = prediction[0]['label']
                confidence = round(prediction[0]['score'], 2)
            else:
                label = "Plant"
                confidence = 0.8

            plant, disease, severity, pesticide, fertilizer = analyze_result(label)

            result = {
                "plant": plant,
                "disease": disease,
                "severity": severity,
                "pesticide": pesticide,
                "fertilizer": fertilizer,
                "confidence": confidence
            }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
