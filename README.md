🚀 Road Condition Classification Using CNN
📌 Overview

This project focuses on classifying road conditions into four categories using Deep Learning techniques. A Convolutional Neural Network (CNN) and a Transfer Learning model (MobileNetV2) were implemented to automatically assess road quality from images.

The system classifies roads into:

✅ Good

⚠️ Satisfactory

❗ Poor

🚨 Very Poor

This solution can assist government authorities and transportation departments in prioritizing road maintenance and improving road safety.

🎯 Project Objectives

Develop an image classification model using CNN.

Implement Transfer Learning using MobileNetV2.

Apply data augmentation to improve model generalization.

Evaluate performance using accuracy and confusion matrix.

Enable deployment for real-world road monitoring systems.

📂 Dataset

Source: Kaggle – Road Damage Classification Dataset

4 Classes: Good, Poor, Satisfactory, Very Poor

Separate Training and Testing folders

Images resized for model input

🧠 Model 1: Custom CNN Architecture
🔹 Input Size

64 × 64 × 3

🔹 Architecture

Conv2D (32 filters) + MaxPooling

Conv2D (64 filters) + MaxPooling

Conv2D (128 filters) + MaxPooling

Dense (64 neurons, ReLU)

Dense (32 neurons, ReLU)

Output Layer (4 neurons, Softmax)

🔹 Training Configuration

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Batch Size: 32

Epochs: 30

🧠 Model 2: MobileNetV2 (Transfer Learning)
🔹 Input Size

224 × 224 × 3

🔹 Approach

Pre-trained MobileNetV2 (without top layers)

Frozen convolutional base

Global Average Pooling

Dense (128 neurons)

Dropout (0.5)

Output Layer (4 neurons, Softmax)

🔹 Training Configuration

Optimizer: Adam (learning rate 0.001)

Loss: Categorical Crossentropy

Batch Size: 16

Epochs: 10

📊 Model Evaluation

Accuracy metric used for performance measurement

Confusion Matrix generated for class-wise analysis

Observed minor misclassifications indicating scope for improvement

💾 Model Deployment

CNN Model saved as: road_damage_model.h5

MobileNetV2 Model saved as: road_condition_model2.h5

Prediction pipeline implemented for testing new images

🛠 Technologies Used

Python

TensorFlow

Keras

NumPy

Matplotlib

ImageDataGenerator

Transfer Learning (MobileNetV2)

📈 Results

Both models demonstrated promising classification accuracy.
The MobileNetV2 model showed improved performance due to transfer learning.

🔮 Future Enhancements

Increase dataset size for better generalization

Fine-tune pre-trained layers

Experiment with ResNet50 / EfficientNet

Deploy as a Web Application

Implement real-time road monitoring system

🌍 Real-World Impact

This system can be integrated with:

Smart City Infrastructure

Automated Road Inspection Systems

Drone-Based Road Monitoring

AI-Based Traffic Safety Systems
