from keras.models import load_model

model = load_model("model/my_model.keras", compile=False)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)