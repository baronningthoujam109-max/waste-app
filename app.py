from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "model/my_model.keras"

# Class names
class_names = ["plastic", "paper", "glass", "metal"]

# Lazy-loaded model
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def predict(img_path):
    model = get_model()

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", result="No file uploaded")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", result="No file selected")

        upload_folder = "static"
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        result = predict(file_path)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)