# Recyclable Material Classifier

A simple but effective deep learning-powered tool to classify everyday materials (plastic, paper, glass, metal, etc.) and predict whether they are recyclable. Built with Python and TensorFlow, it provides an easy-to-use application interface.

## Features
- **Image-based classification** powered by a fine-tuned convolutional neural network.
- **Recyclability indication** based on material type with confidence scoring.
- Lightweight and easy to deploy via a standalone Python app.

## File Structure
Recyclable-Material-Classifier/
├── app.py # Streamlit-powered app for image uploads and predictions
├── RecylableMaterials.py # Model training and fine-tuning script
└── recyclable_material_classifier.h5 # Pre-trained model weights
