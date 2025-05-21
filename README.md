# 🩺 Chest X-ray Disease Classification using ResNet18

This project implements a deep learning model using **PyTorch** and **ResNet18** to classify various diseases from **enhanced chest X-ray images**. It includes data augmentation, training-validation loop, accuracy tracking, and visualization through confusion matrix and classification report.

---

## 📁 Folder Structure

chest-xray-classifier/
├── chest-dataset-enhanced/ # [Ignored] Dataset folder
│ ├── train/
│ └── val/
├── model_training.ipynb # Main Jupyter notebook
├── resnet18_chest.pth # [Optional] Saved trained model
├── .gitignore
└── README.md


---

## 🧠 Model

- Architecture: `ResNet18`
- Framework: `PyTorch`
- Image Size: `224x224`
- Augmentations:
  - Resize
  - Horizontal Flip
  - Random Rotation
  - Color Jitter

---

## 📦 Installation
1. Clone the repository
   
2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. Install dependencies
pip install torch torchvision matplotlib seaborn scikit-learn

4.📂 Dataset
Place your dataset inside a folder named chest-dataset-enhanced/
⚠️ Note: This folder is excluded from Git tracking via .gitignore.

5.🚀 Training
Run the code in model_training.ipynb. It performs:
Training loop with loss and accuracy per epoch
Validation accuracy computation
Model saving (resnet18_chest.pth)
Confusion matrix and classification report

6.📊 Evaluation Output
Classification Report (precision, recall, F1-score)

Confusion Matrix (heatmap)

Optionally: Training Loss and Accuracy plots



