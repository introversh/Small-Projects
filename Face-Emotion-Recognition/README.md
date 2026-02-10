# Real-Time Facial Emotion Recognition

A Python-based project that detects emotions from live webcam feed in real-time using Deep Learning and Computer Vision.

---

## Overview

This project uses a Convolutional Neural Network (CNN) trained on a facial expression dataset to recognize 7 emotions:

**Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**  

The trained model (`model.h5`) is used in combination with OpenCV to detect faces from a webcam and predict emotions live.

---

## Features

- Real-time face detection and emotion recognition
- Trained CNN model with 7 emotion classes
- Demo-ready application for portfolio and internship showcase
- CPU-friendly, works without GPU

---

## Screenshots

![Sample Output](screenshots/sample1.png)  
*Your face with predicted emotion live*

---

## Requirements

- Python 3.x  
- TensorFlow  
- Keras  
- OpenCV  
- NumPy  

Install dependencies with:

```bash
pip install tensorflow keras opencv-python numpy
