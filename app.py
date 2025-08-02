import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Page configuration
st.set_page_config(page_title="Baldness Predictor", layout="centered")
st.title("🧑‍🦲 Baldness Predictor (Model-Based)")

MODEL_PATH = "baldness_cnn_model.h5"


if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Make sure 'baldness_cnn_model.h5' exists in the same folder.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# Inspect input shape and number of channels
input_shape = model.input_shape  
if len(input_shape) != 4:
    st.error(f"Unsupported model input shape: {input_shape}")
    st.stop()
_, height, width, channels = input_shape


class_names = ['bald', 'not_bald']

uploaded_file = st.file_uploader("Upload a front-facing photo of your head", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    image = Image.open(uploaded_file)
    if channels == 1:
        image = image.convert("L")  
    else:
        image = image.convert("RGB")
    image = image.resize((width, height))
    img_array = np.array(image) / 255.0

    
    if channels == 1 and img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    st.caption(f"Image preprocessed shape: {img_array.shape}")

    # Model prediction
    prediction = model.predict(img_array)[0][0]  

    if prediction >= 0.4:
        label = class_names[1]  # not_bald
        emoji = "🧑‍🦱"
        message = f"Low chance of baldness ({(1 - prediction)*100:.1f}%)"
    else:
        label = class_names[0]  # bald
        emoji = "🧑‍🦲"
        message = f"High chance of baldness ({(prediction)*100:.1f}%)"

    st.subheader("Prediction Result:")
    st.success(f"{emoji} {label.upper()} — {message}")

else:
    st.info("Please upload an image to predict.")

# Install required packages with:
# pip install streamlit opencv-python mediapipe numpy pillow tensorflow
