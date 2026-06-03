import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your .keras model
model = tf.keras.models.load_model("my-model.keras")

# CHANGE this according to your dataset classes
classes = ["Organic", "Recyclable", "Non-Recyclable"]

def predict_image(img):
    img = img.resize((224, 224))  # must match training size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return classes[class_index]

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Waste Classification Model"
)

demo.launch()