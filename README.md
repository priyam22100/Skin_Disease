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

### 2. Training with Real Data (HAM10000)
To train with the official HAM10000 dataset:
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Extract all images into the `data/images/` directory.
3. Place `HAM10000_metadata.csv` into the `data/` directory.
4. Run the training script:
```bash
python train_model.py
```
The script will automatically detect the real data, apply class weighting to handle imbalance, and train the model.

### 3. Training with Synthetic Data
If you don't have the real dataset yet, you can generate synthetic images to test the pipeline:
```bash
python generate_data.py
python train_model.py
```

### 3. Run the Web App
Launch the interactive dashboard:
```bash
streamlit run app.py
```

## 🖥️ GPU Support (NVIDIA)
To ensure your NVIDIA GPU is used for training:
1. Install the appropriate NVIDIA drivers for your card.
2. Install CUDA Toolkit and cuDNN.
3. Use the following command to install the GPU-enabled version of TensorFlow (for version 2.16+ it's included in the main package):
```bash
pip install tensorflow[and-cuda]
```
The `train_model.py` script includes a check at the beginning to confirm if the GPU is detected.

## 📊 Evaluation
- **Jupyter Notebook**: Open `skin_disease_prediction.ipynb` for a detailed walkthrough.
- **ROC Curves**: View `roc_curve.png` or check the "Model Performance" tab in the web app.

## ⚠️ Disclaimer
This project is for educational purposes only. It is not intended for clinical use or medical diagnosis. Always consult a professional medical practitioner for skin concerns.
