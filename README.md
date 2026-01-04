# Chest X-Ray Pneumonia Classifier (CNN + Deep CNN + Gradio)

A chest X-ray image classification project that predicts **Pneumonia** vs **Normal** using two TensorFlow/Keras models:
- **Simple CNN**
- **Deep CNN**

It also includes a **Gradio web app** to upload an X-ray image and get predictions from both models.

---

## What this project does

- Loads the **Chest X-Ray Images (Pneumonia)** dataset (train/val/test folders)
- Preprocesses images as:
  - **Grayscale**
  - Resized to **224×224**
  - Normalized to **[0,1]**
- Trains two models (optional in repo / notebook)
- Saves `.h5` models
- Runs a **Gradio interface** for easy testing

---

## Models

### 1) Simple CNN
A small CNN with stacked Conv2D + MaxPooling layers, followed by Dense layers and a sigmoid output.

### 2) Deep CNN
A deeper network with repeated blocks (including **1×1 conv** + **3×3 conv**) + pooling + dropout to reduce overfitting.

Both models output:
- **Normal** (probability ≤ 0.5)
- **Pneumonia** (probability > 0.5)

---

## Dataset

This project expects the dataset in this structure:
```bash
pip install -r requirements.txt
```

requirements.txt:
```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
tensorflow
gradio
```
