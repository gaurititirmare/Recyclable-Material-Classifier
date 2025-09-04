# ♻️ Recyclable Material Classifier
A deep learning–powered web app that classifies materials (plastic, paper, glass, metal, etc.) and predicts whether they are recyclable or not. Built using TensorFlow (MobileNetV2) and deployed with Streamlit.

## 🚀 Features
- Upload an image of a material.
- Predicts the material type using a trained CNN (MobileNetV2 backbone).
- Displays recyclability status (Recyclable / Non-Recyclable).
- Provides confidence score for predictions.
- Interactive Streamlit web interface.

## 🛠️ Tech Stack
- Python 3.8+
- TensorFlow / Keras – Model training & inference
- MobileNetV2 – Pretrained CNN backbone
- Streamlit – Web app UI
- Pillow (PIL) – Image handling
- NumPy – Data preprocessing
- Matplotlib – Training visualization

## 📂 Project Structure
├── app.py                        # Streamlit app for predictions
├── RecylableMaterials.py         # Model training script
├── recyclable_material_classifier.h5   # Trained model
├── class_indices.json            # Class labels mapping
├── dataset/                      # (Not included) Training/Validation/Test dataset
└── README.md                     # Project documentation

## ⚙️ Setup & Installation
1. Clone this repository:
   git clone https://github.com/gaurititirmare/Recyclable-Material-Classifier.git
   cd Recyclable-Material-Classifier

2. Create a virtual environment & activate it:
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

3. Install dependencies:
   pip install -r requirements.txt
   (You can generate requirements.txt with: pip freeze > requirements.txt)

4. Run the Streamlit app:
   streamlit run app.py

## 📊 Training the Model
If you want to retrain the model:
1. Place your dataset inside the dataset/ folder with subfolders:
   dataset/
   ├── train/
   ├── val/
   └── test/
   Each subfolder should have class-wise directories (e.g., plastic/, paper/, etc.).

2. Run the training script:
   python RecylableMaterials.py

3. The trained model will be saved as:
   recyclable_material_classifier.h5

## 📸 Example Usage
- Upload an image of a plastic bottle.
- The app predicts:
  - Material Type: Plastic
  - Recyclable: Yes ♻️
  - Confidence: 96.3%

## 📂 Recommended Datasets
Here are some great datasets you can use to train or retrain this project:
1. Recyclable & Household Waste Classification (~15,000 images)
   https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification
2. Garbage Classification (12 Classes) (15,150 images across 12 categories)
   https://www.kaggle.com/datasets/mostafaabla/garbage-classification
3. DWSD: Dense Waste Segmentation Dataset (784 images, 14 annotated categories)
   https://data.mendeley.com/datasets/gr99ny6b8p/1
4. TrashBox Dataset (~17,785 images, multiple categories)
   https://github.com/AgaMiko/waste-datasets-review
5. Garbage Dataset V2 (~19,762 images, 10 classes)
   https://github.com/AgaMiko/waste-datasets-review
6. RealWaste Dataset (~4,752 real-world images, 9 categories)
   https://github.com/AgaMiko/waste-datasets-review

💡 Start with Kaggle classification datasets for training, then expand with RealWaste or DWSD for real-world robustness.

## 🌍 Motivation
This project promotes sustainability by helping identify recyclable materials using AI. It can be extended to smart waste-management systems, recycling centers, or educational tools.

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repo, create a new branch, and submit a pull request.

## 📜 License
This project is licensed under the MIT License – see the LICENSE file for details.
