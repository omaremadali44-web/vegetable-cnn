import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Vegetable Classifier", page_icon="🥦", layout="centered")
st.title("🥦 Vegetable Image Classifier")
st.write("Upload a vegetable image and let the trained CNN predict its class.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "vegetable_cnn.keras"
CLASS_PATH = BASE_DIR / "class_names.json"
IMG_SIZE = (128, 128)  # Must match the Kaggle notebook.

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return [mapping[str(i)] for i in range(len(mapping))]

if not MODEL_PATH.exists():
    st.error("Model file not found. Copy vegetable_cnn.keras into this folder.")
    st.stop()
if not CLASS_PATH.exists():
    st.error("class_names.json not found. Copy it from Kaggle into this folder.")
    st.stop()

model = load_model()
class_names = load_class_names()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary"):
        resized = image.resize(IMG_SIZE)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=0)

        probabilities = model.predict(array, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        st.subheader(f"Prediction: {class_names[predicted_index]}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.caption("Confidence is the model's predicted probability, not a guarantee that the prediction is correct.")
        top_indices = np.argsort(probabilities)[::-1][:min(5, len(class_names))]
        st.write("Top predictions")
        for idx in top_indices:
            st.write(f"**{class_names[int(idx)]}** — {probabilities[int(idx)]*100:.2f}%")
