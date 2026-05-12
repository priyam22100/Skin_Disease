# Skin Disease Prediction using Deep Learning

This project implements a Deep Learning system to classify skin lesions into 7 different categories. It includes a synthetic data generator (simulating the HAM10000 dataset), a training pipeline, and a Streamlit web application for real-time prediction.

## 🩺 Supported Diseases
The model identifies the following 7 diagnostic categories:
1. **akiec**: Actinic keratoses and intraepithelial carcinoma
2. **bcc**: Basal cell carcinoma
3. **bkl**: Benign keratosis-like lesions
4. **df**: Dermatofibroma
5. **mel**: Melanoma
6. **nv**: Melanocytic nevi
7. **vasc**: Vascular lesions

## 🚀 Features
- **Deep Learning Model**: Convolutional Neural Network (CNN) built with TensorFlow/Keras.
- **High Performance**: Designed to achieve >0.9 Accuracy, Precision, and Recall.
- **Interactive Web App**: Built with Streamlit for easy image uploads and instant results.
- **ROC Curves**: Visualization of model performance for multi-class classification.
- **Portability**: Includes a data generation script to ensure the project works out of the box.

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd skin-disease-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📈 Usage

### 1. Generate Data
Generate the dataset required for training:
```bash
python generate_data.py
```

### 2. Train the Model
Train the CNN and generate evaluation metrics (including ROC curves):
```bash
python train_model.py
```
This produces `skin_disease_model.h5`, `class_names.json`, and `roc_curve.png`.

### 3. Run the Web App
Launch the interactive dashboard:
```bash
streamlit run app.py
```

## 📊 Evaluation
- **Jupyter Notebook**: Open `skin_disease_prediction.ipynb` for a detailed walkthrough.
- **ROC Curves**: View `roc_curve.png` or check the "Model Performance" tab in the web app.

## ⚠️ Disclaimer
This project is for educational purposes only. It is not intended for clinical use or medical diagnosis. Always consult a professional medical practitioner for skin concerns.
