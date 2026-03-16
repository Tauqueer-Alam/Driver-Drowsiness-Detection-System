# 🚗 Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00A69C)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black)

A hybrid, real-time computer vision system to detect driver fatigue and drowsiness. Built specifically to be robust across different lighting conditions and hardware limitations, deploying multiple parallel detection modes. 

This project was built to run effectively on both local machines and web-hosted environments like Hugging Face Spaces.

## 🌟 Key Features

The system offers a unique **Hybrid Detection Architecture**, allowing you to switch between 4 different detection modes on the fly from the web interface:

1. **👀 MediaPipe EAR (Eye Aspect Ratio)**: Uses facial landmarking to calculate the exact distance between the upper and lower eyelids. Fast, very accurate for forward-facing users, but can falter if the face rotates too much.
2. **🧔 Haar Cascades (Classic)**: A classic, lightweight computer vision approach using Haar feature-based cascade classifiers to look for the structural outline of eyes. Excellent for low-end hardware.
3. **🧠 Custom Keras CNN (Deep Learning)**: Cropped images of the driver's eyes are dynamically extracted and fed into a custom-trained Convolutional Neural Network that classifies them as either `Open` or `Closed` in real-time. 
4. **🔥 HYBRID MODE**: Combines all three detection methods simultaneously for maximum robustness and error-checking.

### Web/Mobile Friendly 📱
Includes a modern Flask backend combined with a frontend that correctly provisions the webcam, securely transmitting frames to the server for analysis. Supports ad-hoc HTTPS execution to bypass modern browser restrictions preventing webcam access via standard HTTP.

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Requirements
The application requires specific versions of deep learning frameworks to maintain interoperability. Do not manually upgrade TensorFlow beyond the pinned version.
```bash
pip install -r requirements.txt
```

### 4. Run the Application
The app requires an Ad-hoc SSL Context (`https://`) to allow browsers to access the webcam.
```bash
python app.py
```
*Note: Your browser will warn you that the SSL certificate is "Not Secure". This is normal for local ad-hoc networks; proceed to the site to allow webcam access.*

---

## 🧠 Training Your Own CNN Model

The project includes a separate script `create_cnn_model.py` which designs, builds, and trains the deep learning model used in Mode #3.

If you wish to retrain the model on different datasets:
1. Create a directory named `dataset` in the project root.
2. Inside `dataset`, create two subdirectories: `Open_Eyes` and `Closed_Eyes`.
3. Fill these folders with your new training imagery (square, grayscale crops work best).
4. Run the training script:
```bash
python create_cnn_model.py
```
The script includes automatic real-time data augmentation (rotation, shifting, shearing) and dynamic learning rate reduction to prevent overfitting.

---

## ☁️ Deployment (Hugging Face Spaces)

This repository includes configuration files to run seamlessly on completely headless cloud Docker containers like Hugging Face.

### Critical Deployment Notes:
- `app.py` has been explicitly tuned to dynamically load the model architecture into Keras and map the old `h5` legacy weights physically to bypass common `quantization_config` deserialization errors found in recent TensorFlow distributions.
- **`packages.txt`**: This file contains critical Linux subsystem dependencies (`libgl1-mesa-glx`, `libglib2.0-0`) required by MediaPipe and OpenCV when deploying to environments without a pre-existing X11 graphical layer.

---

## 🛠️ Stack & Technologies
* **Keras / TensorFlow Architecture Configuration**: Convolutional Blocks + Batch Normalization + Max Pooling -> Flatten -> Dense layers with Dropout
* **Haar Cascades**: Open Source frontal face and eye XMLs.
* **MediaPipe API**: Facial Landmarks (`mp.solutions.face_mesh`). 
* **Model Serialization strategy**: Unwrapped weight-loading.

## 🤝 Contributing
Contributions are always welcome. To contribute:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
