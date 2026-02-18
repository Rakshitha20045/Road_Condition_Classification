🚀 Road Condition Classification Using CNN
<p align="center"> <b>Deep Learning-based Road Quality Assessment System</b><br> CNN + Transfer Learning (MobileNetV2) </p>
📌 Overview

This project focuses on classifying road conditions into four categories using Deep Learning techniques.

Two models were implemented:

🧠 Custom Convolutional Neural Network (CNN)

🚀 Transfer Learning using MobileNetV2

The system automatically evaluates road surface quality from images and classifies them into:

✅ Good

⚠️ Satisfactory

❗ Poor

🚨 Very Poor

This solution can assist transportation authorities and smart city systems in prioritizing road maintenance and improving safety.

✨ Key Highlights

Built custom CNN from scratch

Implemented transfer learning using MobileNetV2

Applied data augmentation for better generalization

Evaluated performance using accuracy & confusion matrix

Developed prediction pipeline for real-time inference

Compared base CNN vs pre-trained architecture

🎯 Project Objectives

Develop an image classification model using CNN

Implement Transfer Learning using MobileNetV2

Improve model robustness with data augmentation

Evaluate performance using confusion matrix

Enable scalable deployment for road monitoring systems

📂 Dataset

Source: Kaggle – Road Damage Classification Dataset

Classes: Good, Satisfactory, Poor, Very Poor

Structured into separate training and testing folders

Images resized according to model input requirements

🧠 Model Architectures
🔹 Model 1: Custom CNN
📥 Input Shape

64 × 64 × 3

🏗 Architecture

Conv2D (32 filters) + MaxPooling

Conv2D (64 filters) + MaxPooling

Conv2D (128 filters) + MaxPooling

Dense (64 neurons, ReLU)

Dense (32 neurons, ReLU)

Output Layer (4 neurons, Softmax)

⚙ Training Configuration

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Batch Size: 32

Epochs: 30

🔹 Model 2: MobileNetV2 (Transfer Learning)
📥 Input Shape

224 × 224 × 3

🏗 Approach

Pre-trained MobileNetV2 (without top layers)

Frozen convolutional base

Global Average Pooling

Dense (128 neurons, ReLU)

Dropout (0.5)

Output Layer (4 neurons, Softmax)

⚙ Training Configuration

Optimizer: Adam (learning rate = 0.001)

Loss: Categorical Crossentropy

Batch Size: 16

Epochs: 10

📊 Model Evaluation

Accuracy used as primary performance metric

Confusion Matrix generated for class-wise analysis

Minor misclassifications observed

Transfer Learning model outperformed custom CNN

📈 Results

Both models achieved promising classification performance.

The MobileNetV2 transfer learning model demonstrated improved generalization and higher accuracy due to leveraging pre-trained feature representations.

💾 Model Artifacts

road_damage_model.h5 (Custom CNN)

road_condition_model2.h5 (MobileNetV2)

Prediction pipeline implemented for testing new road images.

🛠 Tech Stack

Python

TensorFlow

Keras

NumPy

Matplotlib

ImageDataGenerator

Transfer Learning (MobileNetV2)

🚀 Deployment Scope

The trained model can be integrated into:

Smart City Infrastructure

Automated Road Inspection Systems

Drone-based Monitoring

AI-powered Traffic Safety Systems

🔮 Future Improvements

Increase dataset size for better generalization

Fine-tune pre-trained MobileNetV2 layers

Experiment with ResNet50 / EfficientNet

Deploy as Web Application

Implement real-time road monitoring

🌍 Real-World Impact

This system enables scalable road condition assessment, reducing manual inspection effort and supporting data-driven infrastructure planning.
