import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/my_model.keras")

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
        path = os.path.join("static", file.filename)
        file.save(path)

        result = predict(path)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)