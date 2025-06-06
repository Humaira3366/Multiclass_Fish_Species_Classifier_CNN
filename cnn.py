from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
import os

# Load the model
model = load_model('cnn_fish_model.h5')

# Fish species class labels
class_names = [
    'animal fish',
    'fish sea_food bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food horse_mackerel',
    'fish sea_food house_mackerel',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# Optional: Add your model accuracy manually if not calculated here
model_accuracy = 0.89  # Example: 89% accuracy

# UI Title and style
st.markdown("<h1 style='text-align: center;'>🐟 Fish Species Classifier with CNN</h1>", unsafe_allow_html=True)
st.markdown("### 📤 Upload a fish image")

# Upload section
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='🖼️ Uploaded Image', use_container_width=True)

    # Image preprocessing
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_label = class_names[np.argmax(prediction)]

    # Output
    st.success(f"🎯 **Predicted Species:** {predicted_label}")
    st.info(f"✅ **Model Accuracy:** {model_accuracy * 100:.2f}%")
