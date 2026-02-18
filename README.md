<p align="center">
  <h1 align="center">Road Condition Classification</h1>
  <h3 align="center">Deep Learning-Based Road Quality Assessment</h3>
</p>

<p align="center">
  <b>CNN • Transfer Learning • Image Classification • AI-Powered</b>
</p>

---

## 📌 Overview

**Road Condition Classification** is a Deep Learning-based image classification system designed to automatically detect and categorize road quality from images.

The system classifies roads into four categories:

- **Good**
- **Satisfactory**
- **Poor**
- **Very Poor**

This solution supports intelligent road maintenance planning and improves transportation safety using AI-driven automation.

---

## 🎯 Objectives

- Build a Convolutional Neural Network (CNN) model.
- Implement Transfer Learning using **MobileNetV2**.
- Improve model generalization using data augmentation.
- Evaluate performance using accuracy and confusion matrix.
- Enable deployment for real-world monitoring systems.

---

## 📂 Dataset

- Source: Kaggle – Road Damage Classification Dataset  
- Total Classes: **4**
  - Good
  - Satisfactory
  - Poor
  - Very Poor
- Separate **Training** and **Testing** directories
- Images resized according to model architecture

---

## 🧠 Model 1 – Custom CNN

### 🔹 Input Shape
`64 x 64 x 3`

### 🔹 Architecture
- Conv2D (32 filters) + MaxPooling  
- Conv2D (64 filters) + MaxPooling  
- Conv2D (128 filters) + MaxPooling  
- Dense (64 neurons, ReLU)  
- Dense (32 neurons, ReLU)  
- Output Layer (4 neurons, Softmax)

### 🔹 Training Configuration
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Batch Size:** 32  
- **Epochs:** 30  

---

## 🚀 Model 2 – MobileNetV2 (Transfer Learning)

### 🔹 Input Shape
`224 x 224 x 3`

### 🔹 Architecture
- Pre-trained **MobileNetV2** (without top layers)
- Global Average Pooling
- Dense (128 neurons, ReLU)
- Dropout (0.5)
- Output Layer (4 neurons, Softmax)

### 🔹 Training Configuration
- **Optimizer:** Adam (Learning Rate: 0.001)
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 16
- **Epochs:** 10

---

## 📊 Model Evaluation

- Performance Metric: **Accuracy**
- Confusion Matrix generated for class-wise analysis
- Minor misclassifications observed (scope for improvement)

---

## 💾 Model Files

- `road_damage_model.h5`
- `road_condition_model2.h5`

---

## 🛠 Tech Stack

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Transfer Learning (MobileNetV2)**

---

## 🔮 Future Improvements

- Increase dataset size for better generalization
- Fine-tune pre-trained layers
- Experiment with ResNet50 / EfficientNet
- Deploy as a web application
- Implement real-time monitoring system
