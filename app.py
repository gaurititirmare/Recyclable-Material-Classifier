import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

# Load the model
model = tf.keras.models.load_model("recyclable_material_classifier.h5")

# Load class indices from JSON file
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse the class indices to get labels from predictions
class_labels = {v: k for k, v in class_indices.items()}

# Define recyclability for each class
recyclability_mapping = {
    "plastic": "Recyclable",
    "paper": "Recyclable",
    "glass": "Recyclable",
    "metal": "Recyclable",
    "other": "Non-Recyclable"  # Example for non-recyclable class
}

# Image size for the model
IMG_SIZE = (224, 224)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize(IMG_SIZE)  # Resize image to model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit App
st.title("Recyclable Material Classifier 🌍")
st.write("Upload an image of a material, and the app will predict the material type and whether it's recyclable.")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(preprocessed_image)
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    recyclability = recyclability_mapping[predicted_class]
    confidence = predictions[0][predicted_index] * 100

    # Display the prediction results
    st.subheader("Prediction Results")
    st.write(f"**Material Type**: {predicted_class.capitalize()}")
    st.write(f"**Recyclable**: {recyclability}")
    st.write(f"**Confidence**: {confidence:.2f}%")

    # Optional: Add additional recyclability information
    if recyclability == "Recyclable":
        st.success("Great! This material is recyclable. Please dispose of it in a recycling bin.")
    else:
        st.error("This material is not recyclable. Please dispose of it responsibly.")
