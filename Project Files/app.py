from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import json
from werkzeug.utils import secure_filename

# Flask app initialization
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for flashing messages

# Upload folder config
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model configuration
IMAGE_SIZE = 224
MODEL_PATH = 'model/rice_model.keras'
LABEL_MAP_PATH = 'model/label_map.json'

# Load trained model
model = load_model(MODEL_PATH)

# Load label map (e.g., { "0": "Arborio", "1": "Basmati", ... })
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

# Convert label_map into sorted list based on index
CATEGORIES = [label_map[str(i)] for i in range(len(label_map))]

# Function to preprocess image and make prediction
def model_predict(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or corrupted.")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_label = CATEGORIES[class_index]
    return predicted_label, confidence

# Route: Home page
@app.route('/')
def home():
    return render_template('index.html')

# Route: Details upload page
@app.route('/details')
def details():
    return render_template('details.html')

# Route: Results page
@app.route('/results')
def results():
    rice_type = request.args.get('type', 'Unknown')
    confidence = request.args.get('confidence', '0')
    image_url = request.args.get('image_url', '')
    return render_template('results.html', type=rice_type, confidence=confidence, image_url=image_url)

# Route: Prediction processing
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "Empty filename", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        label, confidence = model_predict(filepath)
        confidence_percent = round(confidence * 100, 2)
        image_url = f"/static/uploads/{filename}"

        return redirect(url_for('results', type=label, confidence=confidence_percent, image_url=image_url))
    except Exception as e:
        print("Prediction Error:", e)
        return "Prediction failed. Please check the server logs.", 500

# Route: Contact form submission
@app.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')

    # Log the contact form submission
    with open("messages.txt", "a", encoding="utf-8") as f:
        f.write(f"Name: {name}\nEmail: {email}\nSubject: {subject}\nMessage: {message}\n---\n")

    print(f"New Contact Form Submission:\nName: {name}\nEmail: {email}\nSubject: {subject}\nMessage: {message}\n")

    # Flash a success message and redirect to home
    flash("✅ Your message has been sent successfully!", "success")
    return redirect(url_for('home'))

# App run
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
