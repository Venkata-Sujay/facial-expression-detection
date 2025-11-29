# Real-Time Facial Emotion Recognition using Transfer Learning 🎭

A real-time computer vision project that detects faces and classifies emotions into 7 categories using deep learning. This project leverages **Transfer Learning (MobileNetV2)** to achieve high-speed inference suitable for live webcam feeds.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## 📌 Project Overview
The goal of this project was to build a robust system capable of recognizing human facial expressions in real-time. We overcame challenges related to small dataset size and background noise by implementing a custom **Face Cropping Pipeline** and utilizing **Fine-Tuning** techniques on a pre-trained model.

### Supported Emotions:
1.  Happy 😀
2.  Sad ☹️
3.  Anger 😡
4.  Surprise 😲
5.  Fear 😨
6.  Disgust 🤢
7.  Neutral 😐

## ⚙️ Technical Approach

### 1. Data Preprocessing (The Face Cropper)
The raw dataset contained full-body images and noisy backgrounds. I wrote a script using **OpenCV Haar Cascades** to:
* Detect faces in raw images.
* Crop the region of interest (ROI).
* Resize to `224x224` pixels.
* **Result:** This drastically improved model convergence by removing background noise.

### 2. Model Architecture (MobileNetV2)
We used **Transfer Learning** with **MobileNetV2** as the base model.
* **Base:** Pre-trained on ImageNet (frozen initially).
* **Head:** Custom fully connected layers with `Dropout(0.5)` to prevent overfitting.
* **Fine-Tuning:** Unfroze the top layers of MobileNetV2 and retrained with a low learning rate (`1e-5`) to adapt feature extraction to facial nuances.

## 📊 Results & Performance
* **Accuracy:** Achieved ~40% validation accuracy on a 7-class problem (random guessing is ~14%).
* **Generalization:** The model demonstrates strong generalization with Validation Accuracy consistently outperforming Training Accuracy, indicating robust data augmentation.
* **Real-Time Speed:** High FPS on standard CPU due to the lightweight MobileNetV2 architecture.

![Training Graph](path/to/your/accuracy_graph.png)
*(Replace this text with the path to your training graph image)*

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
