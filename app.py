import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Safe model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "my_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["plastic", "paper", "glass", "metal"]

def predict(img_path):
    img = image.load_img(img_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]

        upload_folder = os.path.join("static")
        os.makedirs(upload_folder, exist_ok=True)

        path = os.path.join(upload_folder, file.filename)
        file.save(path)

        result = predict(path)

    return render_template("index.html", result=result)

# IMPORTANT FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)